"""Elaborate the legacy core with pyslang 9.1.0 and an explicit config."""
import argparse
import json
from pathlib import Path
import pyslang

root = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser()
parser.add_argument("--bsg",type=Path,required=True)
args = parser.parse_args()
lock = json.loads((root/"accelerator/locks/basejump.json").read_text())
# Use one compilation unit so the preprocessor definition covers every module.
source = '`define TERNIP_CFG ternip_pkg::RegressionCfg\n'
source += '`include "'+str(args.bsg/"bsg_misc/bsg_defines.sv").replace("\\","/")+'"\n'
import yaml
core = yaml.safe_load((root/"ternip.core").read_text())
paths = [args.bsg/p for p in lock["files"] if not p.endswith("bsg_defines.sv")]
paths += [root/p for p in core["filesets"]["rtl"]["files"] if isinstance(p,str)]
paths += list((root/"rtl/accelerator").glob("*.sv"))
# Inline source to avoid compilation-unit macro ambiguity; includes are resolved
# against the original directory of each file via a preprocessor include path.
for p in paths:
    source += '\n`line 1 "'+str(p).replace("\\","/")+'" 0\n'+p.read_text(encoding="utf-8")+'\n'
options = pyslang.Bag()
pp = pyslang.PreprocessorOptions()
pp.additionalIncludePaths = [str(root/"rtl"),str(args.bsg/"bsg_misc")]
options.preprocessorOptions = pp
co = pyslang.CompilationOptions()
co.topModules = {"ternip_core", "ternip_assertions", "ternip_types_assertions", "ternip_weight_broadcast", "ternip_core_lanes", "ternip_tile_group"}
options.compilationOptions = co
comp = pyslang.Compilation(options)
source_manager = pyslang.SourceManager()
tree = pyslang.SyntaxTree.fromText(source,source_manager,options=options)
comp.addSyntaxTree(tree)
comp.getRoot()
diags = comp.getAllDiagnostics()
engine = pyslang.DiagnosticEngine(comp.sourceManager)
client = pyslang.TextDiagnosticClient()
engine.addClient(client)
for diagnostic in diags: engine.issue(diagnostic)
print(client.getString())
raise SystemExit(1 if engine.numErrors else 0)
