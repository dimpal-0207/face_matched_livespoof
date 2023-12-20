+++++ Face Liveness Detection +++++ 
Use Anti_SPoof model for predict real face and spoof face:
Project Description:


face anti-spoofing model with training architecture, data preprocessing method, model training & test script and open source APK for real time testing.

The main purpose of silent face anti-spoofing detection technology is to judge whether the face in front of the machine is real or fake. The face presented by other media can be defined as false face, including printed paper photos, display screen of electronic products, silicone mask, 3D human image, etc. At present, the mainstream solutions includes cooperative living detection and non cooperative living detection (silent living detection). Cooperative living detection requires the user to complete the specified action according to the prompt, and then carry out the live verification, while the silent live detection directly performs the live verification.

Since the Fourier spectrum can reflect the difference of true and false faces in frequency domain to a certain extent, we adopt a silent living detection method based on the auxiliary supervision of Fourier spectrum. The model architecture consists of the main classification branch and the auxiliary supervision branch of Fourier spectrum



Test Method :
All the test images must be collected by camera, otherwise it does not conform to the normal scene usage specification, and the algorithm effect cannot be guaranteed.
Because the robustness of RGB silent living detection depending on camera model and scene, the actual use experience could be different.
During the test, it should be ensured that a complete face appears in the view, and the rotation angle and vertical direction of the face are less than 30 degrees (in line with the normal face recognition scene), otherwise, the experience will be affected.　



++++Run Project On Local Machine++++
- pip install -r requirements.txt 
+++ run the test file +++
- pyhton3 test.py
- python3 spoofing_face_match_application.py
