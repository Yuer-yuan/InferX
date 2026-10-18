import copy, importlib.util, json, tempfile, unittest
from pathlib import Path
import bind_fresh_caches as b
from unittest import mock
import shutil

class BindingTests(unittest.TestCase):
    def setUp(self):
        self.t=tempfile.TemporaryDirectory(); self.root=Path(self.t.name)
        self.small={'layers':[1], 'file_count':2, 'total_bytes':8, 'files':{'weights-L1-output/packed.bin':{'bytes':4,'sha256':'old'}, 'weights-L1-output/scales.f32':{'bytes':4,'sha256':'old'}}}
        self.full=copy.deepcopy(self.small);self.full['layers']=[0,1]
        for kind in ('packed.bin','scales.f32'):self.full['files']['weights-L0-output/'+kind]={'bytes':4,'sha256':'old'}
        self.make(self.small)
    def tearDown(self):self.t.cleanup()
    def make(self,spec):
        for rel in spec['files']:
            p=self.root/rel;p.parent.mkdir(exist_ok=True);p.write_bytes(b'abcd')
    def test_content_hash_is_new_not_old(self):
        got,origin=b.cache_inventory(self.root,self.small,self.full,True)
        self.assertEqual(got['files']['weights-L1-output/packed.bin']['sha256'], b.digest(self.root/'weights-L1-output/packed.bin'))
        self.assertNotEqual(got['files']['weights-L1-output/packed.bin']['sha256'],'old')
    def test_full_mask_projects_only_selected(self):
        self.make(self.full);got,origin=b.cache_inventory(self.root,self.small,self.full,True)
        self.assertEqual(set(got['files']),set(self.small['files']));self.assertEqual(origin['omitted_serving_files'],2)
    def test_full_mask_not_allowed_for_other_caches(self):
        self.make(self.full)
        with self.assertRaises(ValueError):b.cache_inventory(self.root,self.small,self.full,False)
    def test_unknown_layer_rejected(self):
        (self.root/'weights-L90-output').mkdir()
        with self.assertRaises(ValueError):b.cache_inventory(self.root,self.small,self.full,True)
    def test_missing_file_rejected(self):
        (self.root/'weights-L1-output/packed.bin').unlink()
        with self.assertRaises(ValueError):b.cache_inventory(self.root,self.small,self.full,True)
    def test_unknown_payload_rejected(self):
        (self.root/'weights-L1-output/packed.extra').write_bytes(b'x')
        with self.assertRaises(ValueError):b.cache_inventory(self.root,self.small,self.full,True)
    def test_wrong_size_rejected(self):
        (self.root/'weights-L1-output/scales.f32').write_bytes(b'x')
        with self.assertRaises(ValueError):b.cache_inventory(self.root,self.small,self.full,True)
    def test_metadata_outside_weight_dirs_allowed(self):
        (self.root/'fit-summary.json').write_text('{}');b.cache_inventory(self.root,self.small,self.full,True)
    def test_new_recipe_has_no_old_result_or_receipts(self):
        template={'environment':{'FLAG':'1'},'profiles':{'p':1},'assets':{'cache_evidence':[1]},'measured_results':{'old':1},'selection_status':'passed','quality_status':'passed'};before=copy.deepcopy(template)
        r=b.fresh_recipe(template,'fresh116',{'sha256':'b'*64},'c'*64,'d'*64,{'weights047':{'source':'/fresh'}},'e'*64)
        self.assertEqual(template,before);self.assertEqual(r['environment'],before['environment']);self.assertEqual(r['profiles'],before['profiles'])
        self.assertEqual(r['binary_sha256'],'b'*64);self.assertEqual(r['assets']['contract']['sha256'],'d'*64)
        self.assertNotIn('measured_results',r);self.assertNotIn('cache_evidence',r['assets']);self.assertIn('unmeasured',r['quality_status'])
    def test_unsafe_recipe_id_rejected(self):
        with self.assertRaises(ValueError):b.fresh_recipe({},'../old',{'sha256':'b'*64},'c'*64,'d'*64,{},'e'*64)
    def test_invalid_source_identity_rejected(self):
        with self.assertRaises(ValueError):b.fresh_recipe({},'fresh',{'sha256':'b'*64},'c976','d'*64,{},'e'*64)
    def test_existing_output_cannot_be_overwritten(self):
        with self.assertRaises(ValueError):
            args=['--template-package',str(self.root),'--source-sha256','c'*64,'--recipe-id','fresh','--output',str(self.root)]
            for key in ('weights025','weights041','weights047','binary','model','tokenizer','build-env','eval-envs','tools','modelscope-cache','fit-evidence'):args+=['--'+key,str(self.root)]
            b.main(args)
    def test_known_fit_auxiliaries_are_ignored(self):
        for name in ('expanded.i8','fit-equivalence.json','hessians.json','fit-frozen-before-heldout.json'):
            (self.root/'weights-L1-output'/name).write_bytes(b'aux')
        got,origin=b.cache_inventory(self.root,self.small,self.full,True)
        self.assertEqual(set(got['files']),set(self.small['files']));self.assertEqual(origin['selected_serving_files'],2)
    def test_full_cli_fixture_generates_loadable_isolated_contract(self):
        original=next(candidate for parent in Path(__file__).resolve().parents for candidate in (parent/'entrypoint',parent/'demo/selected100') if (candidate/'candidate_recipes/100-k3-u3-47.json').is_file() and 'qualification=recipe.get("qualification",' in (candidate/'candidate_common.py').read_text())
        package=self.root/'template';shutil.copytree(original,package)
        cat=package/'candidate_recipes';contracts={}
        for name in ['assets-u3-47.json','assets-full64.json']:
            c=json.loads((cat/name).read_text())
            c['model']={'bytes':4,'sha256':b.digest(self.root/'weights-L1-output/packed.bin')}
            for spec in c['caches'].values():
                for item in spec['files'].values():item['bytes']=4
                spec['total_bytes']=len(spec['files'])*4
            (cat/name).write_text(json.dumps(c));contracts[name]=c
        roots={}
        for name,spec in contracts['assets-full64.json']['caches'].items():
            root=self.root/name;roots[name]=root
            for rel in spec['files']:
                dest=root/rel;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(b'abcd')
        model=self.root/'model';model.write_bytes(b'abcd')
        binary=self.root/'serve';binary.write_bytes(b'newbinary');binary.chmod(0o755)
        fit=self.root/'fit.json';fit.write_text('{}')
        output=self.root/'new-deployment';before=b.digest(cat/'100-k3-u3-47.json')
        args=['--template-package',str(package),'--source-sha256','c'*64,'--recipe-id','fresh116','--output',str(output),'--model',str(model),'--binary',str(binary),'--fit-evidence',str(fit)]
        for key,path in roots.items():args+=['--'+key,str(path)]
        for key in ('tokenizer','build-env','eval-envs','tools','modelscope-cache'):args+=['--'+key,str(self.root)]
        with mock.patch.object(b,'TEMPLATE_ASSET_SHA',b.digest(cat/'assets-u3-47.json')),mock.patch.object(b,'TEMPLATE_FULL_SHA',b.digest(cat/'assets-full64.json')):
            b.main(args)
        spec=importlib.util.spec_from_file_location('fresh_fixture_common',output/'entrypoint/candidate_common.py');common=importlib.util.module_from_spec(spec);spec.loader.exec_module(common)
        recipe,recipe_sha=common.load_recipe(output/'entrypoint/candidate_recipes/fresh116.json')
        checked=common.validate_assets(recipe,{'assets':str(output/'assets'),'model':str(output/'assets/model/qwen3_8_27b.ninfer')})
        self.assertEqual(sum(x['file_count']for x in checked['caches'].values()),478)
        self.assertEqual(len(list((output/'assets/weights047').glob('weights-L*'))),47)
        self.assertEqual(recipe['binary_sha256'],b.digest(binary));self.assertEqual(len(recipe['environment']),19)
        self.assertEqual(b.digest(cat/'100-k3-u3-47.json'),before)
        self.assertEqual(set(json.loads((output/'entrypoint/candidate_recipes/index.json').read_text())['recipes']),{'fresh116'})
        self.assertFalse((output/'entrypoint/evidence/C14-r1-summary.json').exists())
if __name__=='__main__':unittest.main()
