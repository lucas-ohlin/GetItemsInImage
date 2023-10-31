import cv2 as cv
import numpy as np
import os

def CountItemsInImage(main_image, item_image):
    #Load main & item image and grayscale them
    main_image = cv.imread(main_image, cv.IMREAD_GRAYSCALE)
    item_image = cv.imread(item_image, cv.IMREAD_GRAYSCALE)

    #Use matchTemplate to find the item in the main image 
    result = cv.matchTemplate(main_image, item_image, cv.TM_CCOEFF_NORMED)
    #If result exceeds this threshold then it's a match
    location = np.where(result >= 0.8)
    
    #Cluster the points that are close to each other
    #This is to avoid repeat rectangles around items
    clusters = []
    for points in list(zip(*location[::-1])):
        for cluster in clusters:
            #Check if the point is close to any existing cluster
            if np.linalg.norm(np.array(points) - np.array(cluster)) <= 5:
                break
        else:
            clusters.append(points)

    #Uncomment this to "see" the images & rectangles around the items for debug purposes
    # w, h = item_image.shape[::-1]
    # for cluster in clusters:
    #     cv.rectangle(main_image, cluster, (cluster[0] + w, cluster[1] + h), (255, 255, 255), 2)
    
    # cv.imshow('Found', main_image)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    
    return len(clusters)

#Get all files in the /Images/ directory 
image_files = [f for f in os.listdir('./Images/') if os.path.isfile(os.path.join('./Images/', f))]
for item_image_name in image_files:
    item_image_path = os.path.join('./Images/', item_image_name)
    print(f"{item_image_name}: {CountItemsInImage('./inputimage.png', item_image_path)}")
