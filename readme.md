Environment setting
============================
``conda env create -f python_38_dvces.yml`` <br>
``conda activate python_38_dvces``<br>
``pip install git+https://github.com/RobustBench/robustbench.git``<br>


Run (MIMIC Counterfactual Generation)
===========================
``python generate_counterfactual_mimic.py --diffusion_ckpt_path /PATH/TO/DIFFUSION/MODEL --robust_classifier_path /PATH/TO/ROBUST/CLASSIFIER/CKPT --mimic_path /PATH/TO/MIMIC/ROOT/DIRECTORY --dicom_id /DICOM/ID/OF/INPUT/IMAGE --labels "{1: 'Edema', 0: 'No Finding'}" --target_class 0``

If you get ``TypeError: 'type' object is not subscriptable``, refer to https://stackoverflow.com/questions/75202610/typeerror-type-object-is-not-subscriptable-python