#%%
import os
import cv2
import pandas as pd
import matplotlib.pyplot as plt

MIMIC_ROOT = '/media/wonjun/HDD8TB/mimic-cxr-jpg-resized512'

#%%
def show_jpgs(
    subject_id = None,
    study_id = None,
    mimic_images_path = '/media/wonjun/HDD8TB/mimic-cxr-jpg-resized512/files'
):
        
    images_path = os.path.join(mimic_images_path,
                            f'p{subject_id}'[:3],
                            f'p{subject_id}',
                            f's{study_id}')
    images = os.listdir(images_path)
    
    if len(images) > 1:
        fig, ax = plt.subplots(1, len(images))
        for i, _img_path in enumerate(images):
            img_path = os.path.join(images_path, _img_path)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ax[i].imshow(img)
        plt.show()
    elif len(images) == 1:
        fig, ax = plt.subplots()
        img_path = os.path.join(images_path, images[0])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.imshow(img)
        plt.show()
    else:
        print("zero images in that path")
        
def show_report(
    subject_id = None,
    study_id = None,
    mimic_reports_path = '/media/wonjun/HDD8TB/mimic-cxr-jpg-resized512/reports/files'
):
    report_path = os.path.join(mimic_reports_path,
                                f'p{subject_id}'[:3],
                                f'p{subject_id}',
                                f's{study_id}.txt')    
    with open(report_path, 'r') as report:
        txt = report.read()
    print(txt)



#%%
p = './mimic-cxr-cardiac-device-labels3.csv'
df = pd.read_csv(p)
df = df.fillna(0)
df
# %% CXRs with both Edema and Cardiac Device
display(df[(df['Edema']==1) & (df['Cardiac Devices']==1)])

#%%
metadata = os.path.join(MIMIC_ROOT, 'mimic-cxr-2.0.0-metadata.csv')
metadata = pd.read_csv(metadata)
metadata
#%%
show_jpgs(subject_id=19997367, study_id=58801152)
row = metadata[metadata['study_id']==58801152]
print(row)