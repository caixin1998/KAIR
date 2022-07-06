echo "Bash version ${BASH_VERSION}..."
degree=8
# iter=290000
# v=train_full_eye_gaze_simple
# s=1
# v=1.3
for i in {00..14}
  do
    # mkdir /home/caixin/GazeData/mpii_448/p$i/face
    #CUDA_VISIBLE_DEVICES=1
    # python inference_gfpgan.py -i /home/caixin/GazeData/MPIIFaceGaze1/Image/p$i/face -o /data1/GazeData/mpii_448/xgaze_512_device0_5000/Image/p$i/face -v train_GFPGAN_xgaze_512_device0 -s 2

    CUDA_VISIBLE_DEVICES=`expr $i % 4` python main_test_face_enhancement.py -i /home/caixin/GazeData/MPIIFaceGaze/Image/p$i/face -o /data1/GazeData/MPIIRes/$1/Image/p$i/face  &

    echo $i
    [ `expr $i` -eq 7 ]  && wait
  done
wait

cp  -r /home/caixin/GazeData/MPIIFaceGaze/Label /data1/GazeData/MPIIRes/$1