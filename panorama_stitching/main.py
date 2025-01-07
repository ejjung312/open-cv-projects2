import os
import cv2

mainFolder = os.path.join(os.getcwd(), 'panorama_stitching', 'data')
myFolders = os.listdir(mainFolder)
print(myFolders)

for folder in myFolders:
    path = os.path.join(mainFolder, folder)
    images = []
    myList = os.listdir(path)
    print(f'Total no of images detected {len(myList)}')
    
    for imgN in myList:
        img_path = os.path.join(path, imgN)
        curImg = cv2.imread(img_path)
        if folder == 2:
            curImg = cv2.resize(curImg, (0,0), None, 0.2, 0.2)
        images.append(curImg)
        
        # print(img_path)
    
    stitcher = cv2.Stitcher.create()
    (status, result) = stitcher.stitch(images)
    
    if status == cv2.STITCHER_OK:
        print('Panorama Generated')
        # img_path = os.path.join(path, result)
        # cv2.imshow(path, result)
        cv2.imshow(folder, result)
        cv2.waitKey(1)
    else:
        print('Panorama Generation Unsuccessful')
        
cv2.waitKey(0)