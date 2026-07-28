 this is a privacy attack red team repo, briefly understand it, and we need to test it on a few method, it should be easy and you only need to understand how to use it and interfaces and no need to know why it works.
Now you need to do this:
1. Download the dataset DigiFace-1M into /raid/wg25r. Select 2000 images as training examples (one image per person) and it will be used across all following methods. 
2. Reproduce the PerceptFace attack already in the repo on this dataset, train it. First validation pass, make sure after training the privacy is protected by measuring sim between faces.  Then during testing, eval on the rest of testings et of DigiFace-1M. The dataset has all images and one image per person, and use 2048 query images that are ANOTHER IMAGE of those person (note that the id overlaps but the image should not, it should use image A of person 1 to search image B of person 1), report top-1 recall, top-0.5% recall, and average rank. 
3. Do the same for https://github.com/ShawnXYang/TIP-IM.git and https://github.com/daizigege/CanFG.git. Note that you might need to retrain these two methods before re-training the FaceLinkGen. 
4. Now repeat everything but with only a limited number of images. The number of images you should try are (100, 200, 500). Try each for the whole run. 


Note: 
1. you should use different family of face embedding models for the protection training and the attack training 
2. You should not get desperate and try to make it work by using different methods or fall back. If it is not working, stop and ask. 
3. User will not monitor your run but will be available if you use the push notif tool to send a message to their phone
4. use only one GPU, put all files in scratchpad first, organize later. 
5. Use a subagent to review before you run. 
6. Ask me questions to clearify things before we start