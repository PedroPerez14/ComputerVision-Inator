#####################################################################################
#
# MRGCV Unizar - Computer vision - Final Course Project
#
# Title: Create a list of images to be matched by Superglue
#
# Date: 12 January 2024
#
#####################################################################################
#
# Authors: Pedro José Pérez García

# Version: 1.0
#
#####################################################################################

import os
import sys

matches_file_folder = '../new_pictures'
matches_file_name = 'matches.txt'
new_images_folder = '../new_pictures'
old_images_folder = '../old_pictures'
old_image_name = 'torre_de_belem_1930_1980.jpg'
superglue_matches_output_folder = '../superglue_output/'

if __name__ == '__main__':

    files = []
    matches = []

    for file in sorted(os.listdir(new_images_folder)):
        if not os.path.isdir(file):
            files.append(file)

    for i in range(len(files)):
        for j in range(len(files)):
            if i != j:
                matches.append(files[i]+" "+files[j]+'\n')
        matches.append(files[i] + " " + os.path.join(old_images_folder, old_image_name)+'\n')
        
    if os.path.exists(os.path.join(matches_file_folder, matches_file_name)):
        os.remove(os.path.join(matches_file_folder, matches_file_name))
       
    f = open(os.path.join(matches_file_folder, matches_file_name), "a")
    for match in matches:
        f.write(match)
    f.close()

    print("Successfully created matches list for superglue!")
    
    os.system("../SuperGluePretrainedNetwork/match_pairs.py --superglue outdoor --max_keypoints 2048 --resize -1 -1 \
              --input_dir " + new_images_folder + " --input_pairs " + os.path.join(matches_file_folder, matches_file_name) + \
                " --output_dir " + superglue_matches_output_folder + " --viz")
    
    print("hehe")