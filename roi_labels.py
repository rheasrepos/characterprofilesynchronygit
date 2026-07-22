import pandas as pd, os
_df=pd.read_csv(os.path.join(os.path.dirname(__file__),'results','roi_labels.csv'))
ROI_NAME={int(r.roi):r['name'] for _,r in _df.iterrows()}
def name_rois(idxs):
    return [(int(i), ROI_NAME.get(int(i),f'ROI#{i}')) for i in idxs]
