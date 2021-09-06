import nibabel as nib
import numpy as np
import argparse
import os, glob, subprocess
from skimage import measure
import pandas as pd

def split_filename(input_path):
    dirname = os.path.dirname(input_path)
    basename = os.path.basename(input_path)

    base_arr = basename.split('.')
    ext = ''
    if len(base_arr) > 1:
        ext = base_arr[-1]
        if ext == 'gz':
            ext = '.'.join(base_arr[-2:])
        ext = '.' + ext
        basename = basename[:-len(ext)]
    return dirname, basename, ext


def purge(dir, pattern):
    for f in glob.glob(os.path.join(dir, pattern)):
        os.remove(f)


def fill_mask_with_coord(mask,coord, value=1): # Default value of 1 to create a binary mask
    for v in range(coord.shape[0]): # Create a loop to iterate over coordinates of all lesion voxels
        x= coord[v,0]
        y = coord[v, 1]
        z = coord[v, 2]
        mask[x,y,z] = value
    return mask

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Rate CV lesions program - Designed by Omar Al-Louzi')
    required = parser.add_argument_group('Required arguments')
    required.add_argument('--images', type=str, nargs='+', required=True,
                        help='Path to FLstar, FLAIR, and T2*-EPI images. Only need 1 example image')
    required.add_argument('--outdir', required=True,
                        help='Output directory where the resultant excel file will be written')
    results = parser.parse_args()

    # Start by printing out some helpful info
    results.images = [os.path.expanduser(image) for image in results.images]
    dir, base, ext = split_filename(results.images[0])
    outdir = os.path.abspath(os.path.expanduser(results.outdir))

    #Test: results=['/home/allouzioa/Python_learning/Test/MID101_t01_20140514_FLstar_MTTE.nii.gz']; dir, base, ext = split_filename(results[0])
    # outdir='/home/allouzioa/Python_learning/Test'

    #Test: results=['/home/allouzioa/Python_learning/Test/CAVS_ID-046_9999999_FLstar_MTTE.nii.gz']; dir, base, ext = split_filename(results[0])
    # outdir='/home/allouzioa/Python_learning/Test'

    id = base[0:19]
    imgs= [os.path.join(dir, id + "_FLstar_MTTE.nii.gz"), os.path.join(dir, id + "_FL_MTTE.nii.gz"), os.path.join(dir, id + "_T2star_MTTE.nii.gz"), os.path.join(dir, id + "_T1_MTTE.nii.gz")]
    print("[INFO]: Now working on analyzing the following images:")
    for i in range(len(imgs)):
        imgs[i] = os.path.abspath(imgs[i])
        print(imgs[i])

    lesion = os.path.join(dir, id + "_CNNLesionMask_MTTE.nii.gz")
    lesion = os.path.abspath(os.path.expanduser(lesion))

    # Now lets work on loading the images
    obj = nib.load(imgs[0]) # Load an example image to calculate dimensions
    origdim = np.asarray(obj.shape,dtype=int) # Extracts length of the np array shape
    vol = np.zeros(obj.shape + (len(imgs),), dtype=np.float32) # Create an empty container for files to be loaded.
    # Added brackets around len function and comma to make it a tuple

    for i, img in enumerate(imgs):
        temp = nib.load(img).get_fdata().astype(np.float32)
        vol[:, :, :, i] = temp
        temp = []

    # Test image views: viewer = napari.view_image(vol[:,:,:,2], name='FLAIR', order=(2,1,0))
    all_lesion = nib.load(lesion).get_fdata().astype(int)
    labels = measure.label(all_lesion, background=0)  # Labels connected regions of an integer array.
    lesion_data = measure.regionprops(labels) # Relatively equivalent to MATLAB's regionprops3 function and measures properties of each lesion cluster

    lesion_data = sorted(lesion_data, key=lambda lesion_data: lesion_data['centroid'][2]) # Sort the lesion cluster list by z-axis coordinate using a lamda special key.

    # propsa = sorted(propsa, key=lambda propsa: propsa['area']) # If you want to change sorting by size
    print('[INFO]: A total of {} lesion(s) detected'.format(len(lesion_data)))
    print()

    # Initialize napari with correct images
    #with napari.gui_qt():
    # %gui qt
    # viewer = napari.view_image(vol[:, :, :, 2], name='T2star', order=(2, 1, 0))
    #viewer = napari.view_image(vol[:, :, :, 2], name='T2star', order=(2, 1, 0), interpolation='lanczos')
    #viewer.add_image(vol[:, :, :, 1], name='FL', interpolation='lanczos')
    #viewer.add_image(vol[:, :, :, 0], name='FLstar', interpolation='lanczos')

    #Initialize a dataframe to collect the results (and eventually be written to excel)
    # c1 = [id] * len(lesion_data) # Generate a column with the id copied over N number of times where N is lesion count
    # c2 = [i+1 for i in range(len(lesion_data))]  # Generate a counter list with N being lesion count # Add 1 to the counter list to adjust for zero indexing
    c1 = []; c2 = []; c3 = []; c4 = []; c5 = []; c6 = []; c7 = []; c8 = []

    start = 1
    temp_excel = os.path.join(outdir, id + '_cvsratings_temp.xlsx')
    if os.path.exists(temp_excel):
        cvs_rating = pd.read_excel(temp_excel)
        c7 = list(cvs_rating.Rater_type)
        c8 = list(cvs_rating.Rater_loc)
        for i in range(1, len(c7)+1):  # Loop over all lesion clusters
            c1.append(id)
            c2.append(i)
            c3.append(lesion_data[i - 1]['area'])  # Add volume info
            # Correct zero indexing coordinate shift
            c4.append(lesion_data[i - 1]['centroid'][0] + 1)  # Add centroid info
            c5.append(lesion_data[i - 1]['centroid'][1] + 1)  # Add centroid info
            c6.append(lesion_data[i - 1]['centroid'][2] + 1)  # Add centroid info
        start = len(c7) + 1
        purge(outdir, id + '_mini*.nii.gz')
        purge(outdir, id + '_lesion_*.nii.gz')

    for i in range(start, len(lesion_data)+1):
        print('[INFO]: Now working on analyzing lesion# {}/{}'.format(i, len(lesion_data)))
        c1.append(id)
        c2.append(i)
        c3.append(lesion_data[i-1]['area']) #Add volume info
        # Correct zero indexing coordinate shift
        c4.append(lesion_data[i-1]['centroid'][0]+1) #Add centroid info
        c5.append(lesion_data[i - 1]['centroid'][1]+1) #Add centroid info
        c6.append(lesion_data[i - 1]['centroid'][2]+1) #Add centroid info

        print('[INFO]: Lesion size: {0:.0f} voxels'.format(c3[i-1]))
        print('[INFO]: Lesion coordinates:')
        print('[INFO]: X -> {0:.0f} ; Y -> {1:.0f} ; Z -> {2:.0f}'.format(c4[i-1],c5[i-1],c6[i-1])) # the {0:.0f} indicates 2 digits of precision and f is used to represent floating point number.

        # Load lesion label to viewer
        bbox_mask = np.zeros(obj.shape, dtype=int)
        pad = 50 # Padding size for mini-image
        xmin = max(0, lesion_data[i-1]['bbox'][0] - pad)
        ymin = max(0, lesion_data[i - 1]['bbox'][1] - pad)
        zmin = max(0, lesion_data[i - 1]['bbox'][2] - pad)

        xmax = min(lesion_data[i-1]['bbox'][3] + pad, vol.shape[0])
        ymax = min(lesion_data[i-1]['bbox'][4] + pad, vol.shape[1])
        zmax = min(lesion_data[i-1]['bbox'][5] + pad, vol.shape[2])
        # bbox_mask[xmin:xmax, ymin:ymax, zmin:zmax ] = 1; # Not needed as I am extracting directly from vols
        # Test image views: viewer = napari.view_image(mini_FLstar, name='Lesion_lbl', order=(2,1,0))
        mini_FLstar = vol[xmin:xmax, ymin:ymax, zmin:zmax, 0]
        temp_FLstar_name = os.path.join(outdir, id + "_miniFLstar_" + str(i) + ".nii.gz")
        nib.Nifti1Image(mini_FLstar, obj.affine, obj.header).to_filename(temp_FLstar_name)

        mini_FL = vol[xmin:xmax, ymin:ymax, zmin:zmax, 1]
        temp_FL_name = os.path.join(outdir, id + "_miniFL_" + str(i) + ".nii.gz")
        nib.Nifti1Image(mini_FL, obj.affine, obj.header).to_filename(temp_FL_name)

        mini_T2star = vol[xmin:xmax, ymin:ymax, zmin:zmax, 2]
        temp_T2star_name = os.path.join(outdir, id + "_miniT2star_" + str(i) + ".nii.gz")
        nib.Nifti1Image(mini_T2star, obj.affine, obj.header).to_filename(temp_T2star_name)

        mini_T1 = vol[xmin:xmax, ymin:ymax, zmin:zmax, 3]
        temp_T1_name = os.path.join(outdir, id + "_miniT1_" + str(i) + ".nii.gz")
        nib.Nifti1Image(mini_T1, obj.affine, obj.header).to_filename(temp_T1_name)

        lesion_vol = np.zeros(obj.shape, dtype=int)
        lesion_vol = fill_mask_with_coord(lesion_vol, lesion_data[i-1].coords, 1)

        mini_lesion = lesion_vol[xmin:xmax, ymin:ymax, zmin:zmax]
        temp_lesion_name = os.path.join(outdir, id + "_lesion_" + str(i) + ".nii.gz")
        nib.Nifti1Image(mini_lesion, obj.affine, obj.header).to_filename(temp_lesion_name)

        # Test image views: viewer = napari.view_image(lesion_vol, name='Lesion_lbl', order=(2,1,0))
        #viewer.add_labels(lesion_vol, name='Lesion_lbl', blending='additive', opacity=0.2, visible=True)

        cmd="itksnap -g " + temp_T1_name + " -o " + temp_T2star_name + " " + temp_FL_name + " " + temp_FLstar_name + " -s " + temp_lesion_name + " 2> /dev/null &";
        proc = subprocess.Popen(cmd, shell=True)
        pid_ = proc.pid
        # print("Process ID is: %d" %pid_)
        # os.system(cmd)

        lesion_type = input("Enter lesion type: ")
        while (type(lesion_type) != int):
            try:
                lesion_type = int(lesion_type)
                if (lesion_type <1) | (lesion_type >7):
                    print("ERROR: Input entered is invalid: '{}'. Select a correct number between 1-6".format(lesion_type))
                    lesion_type = int(input("Enter lesion type: "))
            except ValueError:
                try:
                    lesion_type = float(lesion_type)
                    print("ERROR: Input entered is a float: '{}'. Input should be an integer".format(lesion_type))
                    lesion_type = input("Enter lesion type: ")
                except ValueError:
                    lesion_type = str(lesion_type)
                    print("ERROR: Input entered is a string: '{}'. Input should be an integer".format(lesion_type))
                    lesion_type = input("Enter lesion type: ")

        c7.append(lesion_type)  # Add lesion type info

        lesion_loc = input("Enter lesion location: ")
        while (type(lesion_loc) != int):
            try:
                lesion_loc = int(lesion_loc)
                if (lesion_loc <1) | (lesion_loc >8):
                    print("ERROR: Input entered is invalid: '{}'. Select a correct number between 1-8".format(lesion_loc))
                    lesion_loc = int(input("Enter lesion location: "))
            except ValueError:
                try:
                    lesion_loc = float(lesion_loc)
                    print("ERROR: Input entered is a float: '{}'. Input should be an integer".format(lesion_loc))
                    lesion_loc = input("Enter lesion location: ")
                except ValueError:
                    lesion_loc = str(lesion_loc)
                    print("ERROR: Input entered is a string: '{}'. Input should be an integer".format(lesion_loc))
                    lesion_loc = input("Enter lesion location: ")
        pid_ += 1
        os.system('kill -9 ' + str(pid_))

        c8.append(lesion_loc)  # Add lesion location info
        temp_writer = pd.ExcelWriter(temp_excel)
        df = pd.DataFrame(
            {'SubjID': c1, 'ClusterID': c2, 'Volume': c3, 'x_Centroid_2': c4, 'y_Centroid_1': c5, 'z_Centroid_3': c6,
             'Rater_type': c7, 'Rater_loc': c8})
        df.to_excel(temp_writer, 'Sheet1', index=False)
        temp_writer.save()
        #viewer.layers.remove(3)
        # Delete temporary lesion file
        purge(outdir, id + '_mini*.nii.gz')
        purge(outdir, id + '_lesion_*.nii.gz')
        print()
    print("This is the end of the lesion clusters. Please close the MRI viewer window.")


    print("Writing the multilabel lesion masks and output excel files...")
    # Now work on lesion type data
    type_lesion = np.zeros(obj.shape, dtype=int) # Generate an empty container to create our lesion multilabel masks
    for i in range(1, len(lesion_data) + 1): # Loop over all lesion clusters
        if c7[i-1] !=7:  # Exclude lesions labelled 7 by the rater (artifactual/erroneous lesions that should be deleted)
            fill_mask_with_coord(type_lesion, lesion_data[i - 1].coords, c7[i-1])

    type_outname = os.path.join(outdir, id + "_multilabel_type_mask.nii.gz")
    nib.Nifti1Image(type_lesion, obj.affine, obj.header).to_filename(type_outname)

    # Now work on lesion loc data
    loc_lesion = np.zeros(obj.shape, dtype=int)  # Generate an empty container to create our lesion multilabel masks
    for i in range(1, len(lesion_data) + 1): # Loop over all lesion clusters
        if c7[i - 1] != 7:  # Exclude lesions labelled 7 by the rater (artifactual/erroneous lesions that should be deleted and not included in loc data as well)
            fill_mask_with_coord(loc_lesion, lesion_data[i - 1].coords, c8[i-1])

    loc_outname = os.path.join(outdir, id + "_multilabel_loc_mask.nii.gz")
    nib.Nifti1Image(loc_lesion, obj.affine, obj.header).to_filename(loc_outname)

    # Write our data to excel file
    writer = pd.ExcelWriter(os.path.join(outdir, id + '_cvsratings.xlsx'))
    # writer = pd.ExcelWriter('/path_to_save/output.xlsx')
    df = pd.DataFrame({'SubjID': c1, 'ClusterID': c2, 'Volume': c3, 'x_Centroid_2': c4,  'y_Centroid_1': c5, 'z_Centroid_3': c6, 'Rater_type': c7, 'Rater_loc': c8})
    df.to_excel(writer, 'Sheet1', index=False)
    writer.save()
    os.remove(temp_excel)