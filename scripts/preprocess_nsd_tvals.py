##################################################
# Convert from subject native space to fsaverage #
##################################################

import os
from spacetorch.paths import DS_DIR
from nsdcode.nsd_mapdata import NSDmapdata


nsd = NSDmapdata(DS_DIR)

subjixs = [1, 2, 3, 4, 5, 6, 7, 8]
hemis = ['lh', 'rh']
categories = ['faces', 'places', 'bodies', 'characters', 'objects']

for subjix in subjixs:
    print(f"subject {subjix}")

    for hemi in hemis:
        print(f"\themisphere {hemi}")

        for category in categories:
            print(f"\t\tcategory {category}")

            sourcedata = os.path.join(nsd.base_dir, "nsddata", "freesurfer", f"subj{subjix:02d}", "label", f"{hemi}.floc{category}tval.mgz")

            # nearest-neighbor interpolation to bring the result to fsaverage.
            fsdir = os.path.join(nsd.base_dir, "nsddata", "freesurfer", 'fsaverage')
            nsd.fit(
                subjix,
                f'{hemi}.white',
                'fsaverage',
                sourcedata,
                interptype=None,
                badval=0,
                outputfile=os.path.join(fsdir, "label", f"{hemi}.floc{category}tval_subj{subjix:02d}.mgz"),
                fsdir=fsdir)
