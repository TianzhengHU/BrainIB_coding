import os
import glob
import shutil
import gzip
# Here is make the .ni.gz file together
def grab_nii_files(folder_path):
    # Use glob to find all .nii files in the given folder and its subdirectories
    nii_files = glob.glob(os.path.join(folder_path, '**', '*rest_bold.nii.gz'), recursive=True)
    return nii_files


def copy_and_unzip_files(file_paths, destination_folder):
    # Ensure the destination folder exists
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    for file_path in file_paths:
        if os.path.isfile(file_path) and file_path.endswith('.gz'):
            try:
                # Define the destination path for the unzipped file
                base_name = os.path.basename(file_path)
                destination_path = os.path.join(destination_folder, base_name[:-3])  # Remove '.gz'

                # Unzip and copy the file
                with gzip.open(file_path, 'rb') as f_in:
                    with open(destination_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)

                print(f'Copied and unzipped: {file_path} to {destination_path}')
            except Exception as e:
                print(f'Error copying and unzipping file {file_path}: {e}')
        else:
            print(f'File does not exist or is not a .gz file: {file_path}')


# Example usage
folder_path = '/Users/hutianzheng/Desktop/Brain_IB/DATA/UCLA/ds000030_R1.0.2_sub1_control'
nii_files = grab_nii_files(folder_path)
destination_folder = '/Users/hutianzheng/Desktop/Brain_IB/DATA/UCLA/1_control'
# Print out the list of .nii files
copy_and_unzip_files(nii_files, destination_folder)

# Example usage
folder_path = '/Users/hutianzheng/Desktop/Brain_IB/DATA/UCLA/ds000030_R1.0.2_sub5_schi'
nii_files = grab_nii_files(folder_path)
destination_folder = '/Users/hutianzheng/Desktop/Brain_IB/DATA/UCLA/5_Schi'
# Print out the list of .nii files
copy_and_unzip_files(nii_files, destination_folder)

# Example usage
folder_path = '/Users/hutianzheng/Desktop/Brain_IB/DATA/UCLA/ds000030_R1.0.2_sub6_BIPOLAR'
nii_files = grab_nii_files(folder_path)
destination_folder = '/Users/hutianzheng/Desktop/Brain_IB/DATA/UCLA/6_BIPO'
# Print out the list of .nii files
copy_and_unzip_files(nii_files, destination_folder)

# Example usage
folder_path = '/Users/hutianzheng/Desktop/Brain_IB/DATA/UCLA/ds000030_R1.0.2_sub7_ADHD'
nii_files = grab_nii_files(folder_path)
destination_folder = '/Users/hutianzheng/Desktop/Brain_IB/DATA/UCLA/7_ADHD'
# Print out the list of .nii files
copy_and_unzip_files(nii_files, destination_folder)





