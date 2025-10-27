import numpy as np

def get_avg_speeds(speeds): 
        avg_speeds = []
        for trials in speeds:
            avg_for_type = []
            for trial in trials:
                # Replace 0.0 with NaN, then take mean ignoring NaNs
                cleaned = [np.nan if val == 0.0 else val for val in trial]
                avg_for_type.append(np.nanmean(cleaned))
            avg_speeds.append(avg_for_type)
        return avg_speeds

def get_max_speeds(speeds): 
        max_speeds = []
        for trials in speeds:
            max_for_type = []
            for trial in trials:
                # Replace 0.0 with NaN, then take mean ignoring NaNs
                max_for_type.append(max(trial))
            max_speeds.append(max_for_type)
        return max_speeds
    
def nanmean_per_trial(trials):
    """nanmean per trial; treat 0.0 as missing."""
    return [np.nanmean([np.nan if x == 0.0 else x for x in t]) for t in trials]

def to_id_map(ids, values, cast_int=False):
    """Zip ids->values into a dict, normalising ids to strings (optionally int-like)."""
    out = {}
    for i, v in zip(ids, values):
        key = str(int(float(i))) if cast_int else str(i)
        out[key] = v
    return out

def match_means(baseline_map, event_map):
    """For each baseline id, average all event entries whose key contains that id."""
    out = []
    for bid in baseline_map.keys():
        hits = [val for eid, val in event_map.items() if bid in eid]
        out.append(np.nanmean(hits) if hits else np.nan)
    return out

def clean_pairs(b_means, e_means):
    """Remove pairs where event is NaN."""
    pairs = [(b, e) for b, e in zip(b_means, e_means) if not np.isnan(e)]
    if not pairs:
        return [], []
    b, e = zip(*pairs)
    return list(b), list(e)
