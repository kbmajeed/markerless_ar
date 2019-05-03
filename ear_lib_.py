


"""
Enhanced Augmented Reality (Lib)
"""



import cv2
import numpy as np



"""
Program Launch
"""
#################################################################v##############
def launch():
    print("""
"================================================================="
                        ENHANCED AUGMENTED REALITY
                    Abdulmajeed Muhammad Kabir (2019)
"================================================================="
""")
###############################################################################
    
    


"""
Load Video
"""
###############################################################################
def load_video(path, show=False, **kwargs):
    """
   FUNCTION:
        Call to load video file
     INPUTS:
        path1 = location of video
        kwargs{"resolution"} = output aspect ratio #HD Resolution : 1920×1080 or 1280x720
        kwargs{"volume"} = number of frames to return
    OUTPUTS:
        video frames
    """
#'-----------------------------------------------------------------------------#
    print("Loading Video ...")
    cap = cv2.VideoCapture(path)
    frame_number = 0 
    AllFrames   = []
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            if show:
                cv2.imshow('Video Frames', frame);
                cv2.waitKey(5);
                print("Loading Frame : ", frame_number)
            frame_number += 1
                        
            if kwargs is not None:
                if "volume" in kwargs.keys():
                    volum = kwargs["volume"]
                    AllFrames.append(frame)
                if "resolution" in kwargs.keys():
                    fsize = kwargs["resolution"]
                    AllFrames.append(cv2.resize(frame, fsize))
            else:
                volum = 100000000
                AllFrames.append(frame)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
            if frame_number == volum:
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()

    return AllFrames
###############################################################################
    



"""
Load Image
"""
###############################################################################
def load_advert(path, show=False, **kwargs):
    """
   FUNCTION:
        Call to load Advert
     INPUTS:
        path1 = location of image file
        kwargs{"size"} = output size
    OUTPUTS:
        Advert image
    """
#'-----------------------------------------------------------------------------#
    print("Loading Advert ...")    
    advrt = cv2.imread(path); 
    if "size" in kwargs.keys():
        adsze = kwargs.get("size")
        advrt = cv2.resize(advrt, adsze)
    return advrt
###############################################################################



"""
Image Corners
"""
###############################################################################
def get_corners(image):
    """
   FUNCTION: 
        return the corners of the image
     INPUTS:
        image = source image
    OUTPUTS:
        corners of the source image
    """
#'-----------------------------------------------------------------------------#
    print("Finding Corners of Advert ...")    
    corners = np.zeros((4, 1, 2), dtype=np.float32)
    (height, width) = np.shape(image)[:-1]
    top_left_corner     = [0,     0     ]
    top_right_corner    = [width, 0     ]
    bottom_left_corner  = [0,     height]
    bottom_right_corner = [width, height]
    
    corners[0] = top_left_corner
    corners[1] = top_right_corner
    corners[2] = bottom_left_corner
    corners[3] = bottom_right_corner

    return np.array(corners, dtype='float32')
###############################################################################



"""
Distance Transform Map
"""
###############################################################################
def distance_transform(image):
    """
   FUNCTION: 
        return the distance transform of the image
     INPUTS:
        image = source image
    OUTPUTS:
        distance transform map
    """
#'-----------------------------------------------------------------------------#
    #print("Calculating Distance Transform ...")    
    kernel = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype=np.float32)
    imgL = cv2.filter2D(image, cv2.CV_32F, kernel)
    imgL = np.clip(imgL, 0, 255)
    imgL = np.uint8(imgL)
    sharp = np.float32(image)
    imgR = sharp - imgL
    imgR = np.clip(imgR, 0, 255)
    imgR = imgR.astype('uint8')
    bw   = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(bw, np.max(bw)//2, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    dist = cv2.distanceTransform(bw, cv2.DIST_L2, 3)
    dist = cv2.normalize(dist, dist, 0, 255.0, cv2.NORM_MINMAX)
    return dist
###############################################################################



"""
Scan Map
"""
###############################################################################
def planar_region(image,  method='brute', show=False, **kwargs):
    """
   FUNCTION: 
        return the distance transform of the image
     INPUTS:
        image = source image
        kwargs{window_size} = fixed window size for advert
        kwargs{hstride} = horizontal scanning stride
        kwargs{vstride} = vertical scanning stride  
    OUTPUTS:
        Four corners of the planar region in order: Top left (TL), BL, TR, BR
    """
#'-----------------------------------------------------------------------------#
    print("Marking Starting Point ...")
    if method == 'brute':
        sc = kwargs.get('scale')
        kernel = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype=np.float32)
        image = cv2.filter2D(image, cv2.CV_32F, kernel)
        #cv2.imshow('w',image); cv2.waitKey(0)
        window_size = kwargs.get('window_size')
        hstride = kwargs.get('hstride')
        vstride = kwargs.get('vstride')
        regions = []
        for y in range(0, image.shape[0]-vstride, vstride):
            for x in range(0, image.shape[1]-hstride, hstride):
                region  = image[y:y+window_size[1], x:x+window_size[0]]
                cent    = ((x+x+window_size[0])//2, (y+y+window_size[1])//2)
                score   = np.sum(region)
                dist    = cv2.norm((image.shape[1]//2, image.shape[0]//2),(cent[0], cent[1]))
                regions.append([1.0/(score+1e-10), dist, y, x])
                #print(score, dist, y, x)
                #regions.append([score*dist, y, x])
                if show:
                    ff = cv2.rectangle(image,(x,y),(x+window_size[0],y+window_size[1]),(255,255,255),1)
                    print([x, y, cent, score])
                    cv2.putText(ff, "TL and TR : {}".format((y,x)), cent-15 ,cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    cv2.putText(ff, "Dist : {}".format((int(dist))), cent ,cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    cv2.imshow('window',np.uint8(ff)); 
                    
                    if cv2.waitKey() & 0xFF == ord('q'):
                        cv2.destroyAllWindows()
        regions = np.array(regions)
        regions[:,0] = (regions[:,0]/np.sum(regions[:,0])) 
        regions[:,1] = (regions[:,1]/np.sum(regions[:,1]))
        regions[:,0] = regions[:,0] * regions[:,1]
        regions = regions[regions[:,0].argsort()] 
        select  = regions[-1]
        b = select[2]; 
        a = select[3];

        TL = (a,b); 
        BL = (a+(sc*window_size[0]),b); 
        TR = (a,b+(sc*window_size[1])); 
        BR = (a+(sc*window_size[0]),b+(sc*window_size[1]))
        points = np.array([[TL], [TR], [BL], [BR]], dtype='float32')
        
    elif method == 'smart':
        #img = cv2.imread(path2)
        #img = cv2.resize(img, (640,480))
        #cv2.imshow('w',img); cv2.waitKey(0); cv2.destroyAllWindows()
        sc = kwargs.get('scale')
        imageTransform = distance_transform(image)
        max_region = np.unravel_index(imageTransform.argmax(), imageTransform.shape)
        a0 = max_region[0]
        b0 = max_region[1]
        radius = imageTransform[max_region[0], max_region[1]]
        width = int(np.cos(np.deg2rad(30))*(sc*radius))
        height = int(np.sin(np.deg2rad(30))*(sc*radius))
        
        TL = (b0-width, a0-height); 
        BL = (b0-width, a0+height); 
        TR = (b0+width, a0-height); 
        BR = (b0+width, a0+height);
        if np.sign(a0-height) == -1:
            TL = (b0-width, a0-height+height); 
            BL = (b0-width, a0+height+height); 
            TR = (b0+width, a0-height+height); 
            BR = (b0+width, a0+height+height);
        if np.sign(b0-height) == -1:
            TL = (b0-width+width, a0-height); 
            BL = (b0-width+width, a0+height); 
            TR = (b0+width+width, a0-height); 
            BR = (b0+width+width, a0+height);
        if np.sign(a0-height) == -1 and np.sign(b0-height) == -1:
            TL = (b0-width+width, a0-height+height); 
            BL = (b0-width+width, a0+height+height); 
            TR = (b0+width+width, a0-height+height); 
            BR = (b0+width+width, a0+height+height);            
        #print(TL, BL, TR, BR)
        points = np.array([[TL], [TR], [BL], [BR]], dtype='float32')
        
    else:
        points = np.array([[(0,0)], [(0,100)], [(100,0)], [(100,100)]], dtype='float32')
        
    return points
###############################################################################



"""
Feature-based Image Registration
"""
###############################################################################
def match_frames(image1, image2, method='BFMBased', **kwargs):
    """
   FUNCTION: 
        return the source and destination matches
     INPUTS:
        image1 = source image
        image2 = destination image
        kwargs{num_feats} = number of features to consider
        kwargs{num_select 0.0-1.0} = number of matches to select when using 
            BFSelectBased: if num_select = 1, then BFSelectBased = BFMBased
        methods: FlannBased, BFMBased, BFSelectBased, BFKnnBased
    OUTPUTS:
        source and destination matches
    """
#'-----------------------------------------------------------------------------#
    #print("Performing Feature Based Image Registration ...")
    num_feats = kwargs.get('num_feats')
    num_select = kwargs.get('num_select')
    
    if method == 'FlannBased':
        orb             =  cv2.ORB_create(num_feats)
        kp1, des1       =  orb.detectAndCompute(image1, None)
        kp2, des2       =  orb.detectAndCompute(image2, None)
        FLANN_INDEX_LSH =  6
        index_params    =  dict(algorithm = FLANN_INDEX_LSH, table_number = 6, key_size = 10, multi_probe_level = 1) #6,10,3
        search_params   =  dict(checks = 150)
        flann           =  cv2.FlannBasedMatcher(index_params, search_params)
        matches         =  flann.knnMatch(des1,des2,k=2)
        MIN_MATCH_COUNT =  10
        good = []
        for m,n in matches:
            if m.distance < 0.75*n.distance:
                good.append(m)
        if len(good)>MIN_MATCH_COUNT:
            src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
            dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
        else:
            print("Insufficient Matches")
        return src_pts, dst_pts
    
    elif method == 'BFMBased':
        orb        =  cv2.ORB_create(num_feats)
        kp1, des1  =  orb.detectAndCompute(image1, None)
        kp2, des2  =  orb.detectAndCompute(image2, None)
        bf      = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True) 
        matches = bf.match(des1,des2)
        matches = sorted(matches, key = lambda x:x.distance)
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in matches ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in matches ]).reshape(-1,1,2)
        return src_pts, dst_pts
    
    elif method == 'BFSelectBased':
        orb        =  cv2.ORB_create(num_feats)
        kp1, des1  =  orb.detectAndCompute(image1, None)
        kp2, des2  =  orb.detectAndCompute(image2, None)
        bf      = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True) 
        matches = bf.match(des1,des2)
        matches = sorted(matches, key = lambda x:x.distance)
        #TAKING FIRST K in SORTED MATCHES
        matches = matches[0:int(num_select*len(matches))]
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in matches ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in matches ]).reshape(-1,1,2)
        return src_pts, dst_pts

    elif method == 'BFKnnBased':
        orb       =  cv2.ORB_create(num_feats)
        kp1, des1 = orb.detectAndCompute(image1,None)
        kp2, des2 = orb.detectAndCompute(image2,None)
        bf        = cv2.BFMatcher(cv2.NORM_HAMMING)
        matches   = bf.knnMatch(des1,des2, k=2)
        good = []
        for m,n in matches:
            if m.distance < 0.9*n.distance:
            #if m.distance < 0.75*n.distance:
                good.append(m)
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
        return src_pts, dst_pts        
        
###############################################################################



"""
Aesthetics to Advert
"""
###############################################################################
def advert_modify(image, mag=1, space=50):
    """
   FUNCTION: 
        applies some edge feathering processing to advert image
     INPUTS:
        image = advert image
    OUTPUTS:
        processed advert image
    """
#'-----------------------------------------------------------------------------#
    frame1 = np.zeros_like(image)
    frame2 = np.copy(image)
    Ho, Wo = np.shape(frame2)[:-1]
    inner = image[space:Ho-space, space:Wo-space]
    frame1[space:Ho-space, space:Wo-space] = inner   
    frame2[space:Ho-space, space:Wo-space] = 255
    frame2 = cv2.GaussianBlur(frame2, (int(mag*33),int(mag*33)), int(mag*15))
    frame2[space:Ho-space, space:Wo-space] = inner
    #frame2 = cv2.flip(frame2, +1)
    return frame2
###############################################################################



