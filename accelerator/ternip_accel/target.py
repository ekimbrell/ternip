"""Explicit board placement and memory allocation; no bank/SLR inference."""
import copy
import itertools
import math
import re
from .compiler import canonical, digest, align

PC_BYTES = 256*1024**2

def memory_indices(tag):
    match = re.fullmatch(r"(DDR|HBM)\[(\d+)(?::(\d+))?\]", tag)
    if not match: raise ValueError(f"Invalid memory tag {tag}")
    kind, first, last = match.groups()
    first, last = int(first), int(last or first)
    if first > last or first < 0 or last >= (32 if kind == "HBM" else 2):
        raise ValueError("Memory range exceeds U280 topology")
    return kind, list(range(first,last+1))

def validate(target):
    t = copy.deepcopy(target)
    if t.get("version") != 1 or t.get("board") != "u280": raise ValueError("Expected U280 target v1")
    if t.get("backend", "hls-reference") not in ("hls-reference", "rtl-tiles"): raise ValueError("Unknown compute backend")
    if not t.get("platform"): raise ValueError("An explicit platform is required")
    if t.get("clock_mhz") not in (200,250,300): raise ValueError("Unsupported clock")
    if t.get("core_batch_size") != 1: raise ValueError("Each legacy core must use BatchSize=1")
    if t.get("lanes_per_group") not in (1,2,4,8,16): raise ValueError("Invalid lane count")
    if t.get("ternary_parallelism") not in (128,256) or t.get("vector_parallelism") not in (4,8,16):
        raise ValueError("Invalid parallelism")
    groups = t.get("groups",[])
    if len(groups) not in (1,2,4,8,16) or len(groups)*t["lanes_per_group"] > 64:
        raise ValueError("Invalid group count or more than 64 lanes")
    names, weight_banks = set(), set()
    hbm_owners = {}
    for g in groups:
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*",g["cu"]) or g["cu"] in names:
            raise ValueError("Invalid or duplicate CU")
        names.add(g["cu"])
        if g["slr"] not in (0,1,2): raise ValueError("U280 only has SLR0-2")
        if set(g["ports"]) != {"weights","state","control"}: raise ValueError("Missing AXI assignment")
        for tag in g["ports"].values(): memory_indices(tag)
        kind, banks = memory_indices(g["ports"]["weights"])
        keys = {(kind,b) for b in banks}
        if keys & weight_banks: raise ValueError("Groups must have disjoint weight banks")
        weight_banks |= keys
        for port,tag in g['ports'].items():
            kind,banks=memory_indices(tag)
            if kind!='HBM': continue
            role='weights' if port=='weights' else 'state_control'
            owner=(g['cu'],role,tag)
            for bank in banks:
                if bank in hbm_owners and hbm_owners[bank]!=owner:
                    raise ValueError('HBM allocations overlap between groups or weight/state traffic')
                hbm_owners[bank]=owner
    if t.get('backend')=='rtl-tiles' and any(name+'_tiles' in names for name in names):
        raise ValueError('Controller and RTL CU names collide')
    t["build_hash"] = digest(canonical({k:v for k,v in t.items() if k != "build_hash"}))
    return t

def link_config(target):
    t = validate(target)
    names = ":".join(g["cu"] for g in t["groups"])
    lines = ["[connectivity]",f"nk=ternip_group:{len(t['groups'])}:{names}"]
    rtl=t.get("backend")=="rtl-tiles"
    if rtl:
        tile_names=":".join(g["cu"]+"_tiles" for g in t["groups"])
        if any(g["cu"]+"_tiles" in {x["cu"] for x in t["groups"]} for g in t["groups"]):
            raise ValueError("Controller and RTL CU names collide")
        lines.append(f"nk=ternip_rtl:{len(t['groups'])}:{tile_names}")
    for g in t["groups"]:
        if rtl:
            tile=g["cu"]+"_tiles"
            lines += [f"slr={tile}:SLR{g['slr']}",
                      f"stream_connect={g['cu']}.tile_command:{tile}.command:32",
                      f"stream_connect={tile}.response:{g['cu']}.tile_response:32"]
        lines.append(f"slr={g['cu']}:SLR{g['slr']}")
        for port, tag in sorted(g["ports"].items()):
            args = {"weights":["weights"], "state":["state","scratch"], "control":["control","program","parameters","counters"]}[port]
            for arg in args: lines.append(f"sp={g['cu']}.{arg}:{tag}")
    lines += ["[clock]"] + [f"freqHz={t['clock_mhz']*1000000}:{g['cu']}" for g in t["groups"]]
    if rtl: lines += [f"freqHz={t['clock_mhz']*1000000}:{g['cu']}_tiles" for g in t["groups"]]
    return "\n".join(lines)+"\n"

def footprint(manifest, lanes, queue_bytes_per_lane=65536):
    a = manifest["allocation"]
    return {"weights":align(a["weights_bytes"]),
            "state":align(lanes*(a["state_bytes_per_lane"]+a["scratch_bytes_per_lane"])),
            "control":align(a["parameters_bytes"]+a["program_bytes"]+lanes*queue_bytes_per_lane)}

def assign_hbm(target, manifest):
    t = validate(target)
    sizes = footprint(manifest,t["lanes_per_group"])
    bank = 0
    for g in t["groups"]:
        for ports, size in ((["weights"], sizes["weights"]),(["state","control"],sizes["state"]+sizes["control"])):
            n = max(1,math.ceil(size/PC_BYTES))
            if bank+n > 32: raise ValueError("Replicated model and lane allocations exceed HBM capacity/topology")
            tag = f"HBM[{bank}]" if n == 1 else f"HBM[{bank}:{bank+n-1}]"
            for port in ports: g["ports"][port] = tag
            bank += n
    # State/control share capacity but have an AXI path separate from weights.
    return validate(t)

def select_device(devices, board="u280", identifier=None):
    matches = [d for d in devices if re.search(r"(?:^|_)"+re.escape(board)+r"(?:_|$)",d["name"].lower())
               and (identifier is None or str(d["id"]) == str(identifier))]
    if len(matches) != 1: raise ValueError(f"Expected exactly one {board}; found {len(matches)}")
    return matches[0]

def resolve_memory(metadata, tag, bytes_required, grouped_verified=False):
    kind, banks = memory_indices(tag)
    if len(banks)>1 and not grouped_verified:
        raise ValueError("Grouped XRT allocation has not been verified for this stack")
    matches = [m for m in metadata if m["tag"] == tag and m.get("used", True)]
    if len(matches) != 1: raise ValueError(f"xclbin does not expose exactly one allocation for {tag}")
    m = matches[0]
    if bytes_required > m["size_bytes"]: raise ValueError(f"Allocation exceeds {tag} capacity")
    return m["group_id"]

def sweep(base, manifest):
    for groups, lanes, tp, vp, clock in itertools.product((1,2,4,8,16),(1,2,4,8,16),(128,256),(4,8,16),(200,250,300)):
        if groups*lanes > 64: continue
        t = copy.deepcopy(base)
        t.update(lanes_per_group=lanes, ternary_parallelism=tp, vector_parallelism=vp,clock_mhz=clock)
        t["groups"] = [{"cu":f"ternip_group_{i}","slr":i%3,
                        "ports":{"weights":f"HBM[{2*i}]","state":f"HBM[{2*i+1}]","control":f"HBM[{2*i+1}]"}} for i in range(groups)]
        try: yield assign_hbm(t,manifest)
        except ValueError: continue
