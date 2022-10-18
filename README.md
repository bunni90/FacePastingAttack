# Face pasting Attack

By Niklas Bunzel and Lukas Graner

This is the code of our entry in the [MLSec face recognition challenge](https://mlsec.io/) and the corresponding [article](https://arxiv.org/abs/2210.09153).

The challenge is organized by Adversa AI, Cujo AI and Robust Intelligence. The goal was to attack a black box face recognition model with targeted attacks. 
The model returned the confidence of the target class and a stealthiness score. For an attack to be considered successful the target class has to have the highest confidence among all classes and the stealthiness has to be at least 0.5. 
In our approach we paste the face of a target into a source image. By utilizing position, scaling, rotation and transparency attributes we reached 3rd place. 
Our approach took approximately 200 queries per attack for the final highest score and about ~7.7 queries minimum for a successful attack.

![An example of a successfull attack.](0_1.png)

## The repo

* facerecognitionsamples: Contains the images given by Adversa AI
* facemasks_manual: Contains our manually created facemasks for the usage with `manual_face_pasting_attack.ipynb`
* `model.py` & `resnet.py`: The BiseNet implementation from [face-parsing.PyTorch](https://github.com/zllrunning/face-parsing.PyTorch)
* `manual_face_pasting_attack.ipynb` & `face_pasting_attack.py`: Our attack code for the manual face masks and automatic face masks based on BiseNet respectivley

In order to run the attacks please set the file paths and API keys accordingly and download the weights for the BiseNet from [face-parsing.PyTorch](https://drive.google.com/open?id=154JgKpzCPW82qINcVieuPH3fZ2e0P812).

For the usage of the manual masks run the notebook `manual_face_pasting_attack.ipynb` otherwise run `python face_pasting_attack.py`.
The results will be written to jsonl files, each line representing one query with success, confidence, stealthiness and attack parameters.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.