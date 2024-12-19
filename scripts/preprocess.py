import os
import numpy as np
import pandas as pd
from spym.io import omicronscala
from tqdm import tqdm


def main():
    # Paths
    raw_data_path = os.path.join("data", "raw_data")
    metadata_path = os.path.join(raw_data_path, "metadata.csv")
    stm_data_path = os.path.join(raw_data_path, "STM", "STM", "data")
    output_path = os.path.join("data", "filled_empty", "stm_data.npy")
    
    # Read metadata
    df = pd.read_csv(metadata_path)
    # Filter rows where FieldXSizeinnm == 100
    df_400 = df[(df["FieldXSizeinnm"] == 100)&(df['ImageSizeinX'] == 400)&(df['ImageSizeinY'] == 400)]
    df_384 = df[(df["FieldXSizeinnm"] == 100)&(df['ImageSizeinX'] == 384)&(df['ImageSizeinY'] == 384)]
    print(f"Found {len(df_400)+len(df_384)} images with FieldXSizeinnm == 100")

    patch_size = 128
    num_patches_side = 3
    
    images_patches = []
    images_labels = []
    for df in [df_400, df_384]:
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing images"):
            image_id = str(row["ID"])  
            img_path = os.path.join(stm_data_path, f"{image_id}.par")
            img_data = omicronscala.to_dataset(img_path)
            filled = np.array(img_data.Topography_Forward)
            if img_data.Topography_Backward is None:
                empty = np.zeros_like(filled)
            else:
                empty = np.array(img_data.Topography_Backward)
        

            # If the image is already 400x400, we just consider the top-left 384x384 section.
            H, W = filled.shape
            if H < 384 or W < 384:
                # If the image is smaller than expected, skip or handle differently
                continue
            
            # Crop to 384x384
            filled_cropped = filled[0:384, 0:384]
            empty_cropped = empty[0:384, 0:384]

            # Now create the 9 patches of 128x128 in row-major order
            for i in range(num_patches_side):
                for j in range(num_patches_side):
                    f_patch = filled_cropped[i*patch_size:(i+1)*patch_size,
                                             j*patch_size:(j+1)*patch_size]
                    e_patch = empty_cropped[i*patch_size:(i+1)*patch_size,
                                            j*patch_size:(j+1)*patch_size]
                    patch_stack = np.stack([f_patch, e_patch], axis=0)
                    images_patches.append(patch_stack)
                    images_labels.append(f'{image_id}_{i}_{j}')

    images_labels = np.array(images_labels)
    np.save(output_path.replace('.npy', '_labels.npy'), images_labels)
    
    # Convert to a numpy array of shape (N_images*9, 2, 128, 128)
    if len(images_patches) > 0:
        images_array = np.stack(images_patches, axis=0)
        np.save(output_path, images_array)
        print(f"Saved {len(images_patches)} patches to {output_path}")
    else:
        print("No images found with the specified criteria.")

    # save smaller train / test set for hackathon
    imgs = np.load('data/filled_empty/stm_data.npy')
    labels = np.load('data/filled_empty/stm_data_labels.npy')
    train_imgs = imgs[:1200,0,:,:]
    test_imgs = imgs[1200:1400,0,:,:]
    train_labels = labels[:1200]
    test_labels = labels[1200:1400]

    # split train to 2 files for github
    np.save('data/filled_empty/train_imgs_0.npy', train_imgs[:600])
    np.save('data/filled_empty/train_imgs_1.npy', train_imgs[600:])
    np.save('data/filled_empty/test_imgs.npy', test_imgs)
    np.save('data/filled_empty/train_labels.npy', train_labels)
    np.save('data/filled_empty/test_labels.npy', test_labels)

if __name__ == "__main__":
    main()
