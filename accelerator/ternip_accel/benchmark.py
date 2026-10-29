"""Host-observed warm-resident measurements; qualification is a separate step."""
import statistics
import time

def measure(runtime,prompts,output_tokens=64,quantum=8,repetitions=5):
    if repetitions<3 or not prompts or output_tokens<2: raise ValueError('Need >=3 repetitions and >=2 output tokens')
    reports=[]
    for repeat in range(repetitions):
        runtime.reset();ids=[f'bench-{repeat}-{i}' for i in range(len(prompts))]
        counts={key:0 for key in ids};first={};arrivals={key:[] for key in ids};done=set()
        start=time.perf_counter();decode_start=None;decode_base=0;decode_end=None;decode_count=0
        for key,prompt in zip(ids,prompts): runtime.submit(key,prompt,output_tokens)
        while len(done)<len(ids):
            runtime.advance(quantum)
            now=time.perf_counter()
            for key in ids:
                tokens,status=runtime.collect(key)
                if tokens:
                    first.setdefault(key,now-start)
                    arrivals[key].extend([now-start]*len(tokens));counts[key]+=len(tokens)
                if status=='error': raise RuntimeError('Device error during benchmark')
                if status in ('complete','cancelled','reset'): done.add(key)
            if len(first)==len(ids) and decode_start is None:
                decode_start=now;decode_base=sum(counts.values())
            elif decode_start is not None and done and decode_end is None:
                decode_end=now;decode_count=sum(counts.values())-decode_base
        elapsed=time.perf_counter()-start
        reports.append(dict(warm_resident=True,model_loading_included=False,quantum=quantum,
            generated_tokens=sum(counts.values()),elapsed_s=elapsed,
            workload_tokens_per_second=sum(counts.values())/elapsed,
            steady_decode_tokens_per_second=(decode_count/(decode_end-decode_start) if decode_end and decode_end>decode_start else None),
            ttft_s=first,token_delivery_times_s=arrivals,
            per_request_mean_token_latency_s={key:(times[-1]-times[0])/(len(times)-1) if len(times)>1 else None for key,times in arrivals.items()},
            latency_resolution='host collection at quantum boundaries',
            measured_lane_occupancy=runtime.counters() if hasattr(runtime,'counters') else None))
    return dict(status='measurement-not-qualification',repetitions=reports,
        median_workload_tokens_per_second=statistics.median(r['workload_tokens_per_second'] for r in reports))

def qualified_winner(reports):
    eligible=[]
    for r in reports:
        evidence=r.get('qualification',{})
        if not all(evidence.get(k) is True for k in ('correctness','quality','routed_setup','routed_hold','hardware')): continue
        if not all(r.get(k) is not None for k in ('clock_mhz','temperature_c','topology','version_hashes')): continue
        samples=r.get('steady_decode_tokens_per_second',[])
        if len(samples)<3 or any(x<=0 for x in samples): continue
        mean=statistics.mean(samples);ci=1.96*statistics.stdev(samples)/(len(samples)**.5)
        eligible.append((mean,ci,r))
    if not eligible: return None
    best=max(eligible,key=lambda v:v[0])
    # Approximate 95% normal confidence intervals; overlapping intervals are
    # treated as indistinguishable, then latency and normalized resource usage.
    tied=[r for mean,ci,r in eligible if mean+ci>=best[0]-best[1]]
    return min(tied,key=lambda r:(r['mean_request_latency_s'],r['normalized_resource_usage']))
