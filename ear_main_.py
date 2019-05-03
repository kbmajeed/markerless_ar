

"""
Enhanced Augmented Reality (Main)
"""
import os
import cv2
import time
import ear_lib_
ear_lib.launch()


advert_dir = "input_advert"
input_dir  = "input_videos"
output_dir = "output_videos"


"GOOD VIDEO DATASET PATHS"
path1a = os.path.join(input_dir, 'Test_Case1.mp4')


"ADVERT IMAGE DATASET PATHS"
path2a = os.path.join(advert_dir, 'hp_logo.png')


def EAR_TEST_CASE1(path1=path1a, path2=path2a, path3=output_dir):
  print("\n=======================INITIALIZATION==========================")     
  Frames = ear_lib.load_video(path1, volume=9999)
  advert = ear_lib.load_advert(path2)
  advert = ear_lib.advert_modify(advert, mag=1, space=5)
  origin_points   = ear_lib.get_corners(advert)
  marker_points   = ear_lib.planar_region(Frames[0], method='smart', show=False, scale=0.8)
  fWidth, fHeight = Frames[0].shape[:-1][::-1]
  save_video = cv2.VideoWriter(os.path.join(path3, 'Output_Test_Case1.mp4'), cv2.VideoWriter_fourcc(*'MP4V'), 30, (fWidth, fHeight))
  print("\n============PERFORM ENHANCED AUGMENTED REALITY================")
  Tic = time.time()
  for i in range(len(Frames)-1):
          src_pts, dst_pts = ear_lib.match_frames(Frames[i], Frames[i+1], method='BFKnnBased', num_feats=1800)
          HH1, mask1       = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0); HH1 = HH1/HH1[2,2]
          marker_points    = cv2.perspectiveTransform(marker_points, HH1)
          HH2              = cv2.getPerspectiveTransform(origin_points, marker_points); HH2 = HH2/HH2[2,2]
          warped_frame     = cv2.warpPerspective(advert, HH2, Frames[i+1].shape[:-1][::-1], Frames[i+1], flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_TRANSPARENT)
          save_video.write(warped_frame)
  Toc = time.time()
  save_video.release()
  print("\n========================COMPLETE=============================")
  print("Total Frames  : ", len(Frames))
  print("Time Elapsed  : ", (Toc-Tic))
  print("Estimated FPS : ", len(Frames)/float((Toc-Tic)))
  del Frames  

  
if __name__ == '__main__':
    EAR_TEST_CASE1(path1a, path2a) #ad in vid

    






































