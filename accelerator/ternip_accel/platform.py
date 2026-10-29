"""Conservative platforminfo evidence validation; unknown schemas fail closed.

Record exact platform metadata alongside every build. These checks establish
advertised topology, not routed availability, timing closure, or XRT compatibility.
"""
import re
from .target import validate,memory_indices


def inspect_platform(info,target):
    t=validate(target);parts=set();memories=set();slrs=set();clocks=[]
    def visit(value,path=()):
        if isinstance(value,dict):
            unit=str(value.get('unit',value.get('units',''))).lower()
            if unit in ('hz','khz','mhz'):
                for field in ('frequency','freq'):
                    if isinstance(value.get(field),(int,float)):
                        hz=float(value[field])*{'hz':1,'khz':1000,'mhz':1000000}[unit]
                        clocks.append(dict(path='.'.join(path+(field,)),frequency_hz=hz))
            for key,child in value.items():visit(child,path+(str(key),))
        elif isinstance(value,list):
            for i,child in enumerate(value):visit(child,path+(str(i),))
        elif isinstance(value,str):
            if re.fullmatch(r'xcu\d+-[a-z0-9-]+',value,re.I):parts.add(value.lower())
            if re.fullmatch(r'(DDR|HBM)\[\d+(?::\d+)?\]',value):
                kind,banks=memory_indices(value)
                memories.update(f'{kind}[{bank}]' for bank in banks)
            if re.fullmatch(r'SLR[0-9]+',value,re.I):slrs.add(int(value[3:]))
        elif isinstance(value,(int,float)) and not isinstance(value,bool) and path:
            key=re.sub('[^a-z]','',path[-1].lower())
            if key in ('slr','slrid','slrindex') and int(value)==value and value>=0:slrs.add(int(value))
            if key in ('numslrs','slrcount','numberofslrs') and int(value)==value and 0<value<100:
                slrs.update(range(int(value)))
            for suffix,scale in (('mhz',1000000),('khz',1000),('hz',1)):
                if key in ('freq'+suffix,'frequency'+suffix,'clockfrequency'+suffix):
                    clocks.append(dict(path='.'.join(path),frequency_hz=float(value)*scale))
                    break
    visit(info)
    if parts!={t['part'].lower()}:raise ValueError(f'Platform part evidence missing, ambiguous, or mismatched: {sorted(parts)}')
    if not slrs or not slrs.issubset({0,1,2}):raise ValueError('Missing or invalid U280 SLR metadata')
    if not clocks or any(c['frequency_hz']<=0 for c in clocks):raise ValueError('Missing explicit clock-frequency units in platforminfo')
    if t['clock_mhz']*1000000>max(c['frequency_hz'] for c in clocks):raise ValueError('Requested clock exceeds all advertised platform clock frequencies')
    for group in t['groups']:
        if group['slr'] not in slrs:raise ValueError(f"Platform does not advertise SLR{group['slr']}")
        for tag in group['ports'].values():
            kind,banks=memory_indices(tag)
            missing={f'{kind}[{bank}]' for bank in banks}-memories
            if missing:raise ValueError(f'Platform memory tags missing: {sorted(missing)}')
    return dict(part=t['part'],memory_tags=sorted(memories),slrs=sorted(slrs),clocks=clocks,
        requested_clock_hz=t['clock_mhz']*1000000,
        status='advertised-topology-checked-not-routed-or-hardware-qualified')
