import pandas as pd
import numpy as np

# annotation file that contains the video id, toa, and the risky objects traking number
anno_file = 'm_set1_frames.csv'
# deepsort generated text files directory (video tracking information stored in this text file)
track_text = 'track_text/m_set1_frames/'


vid_ids = []

with open(anno_file, 'r') as f:
    for line in f.readlines():
        a = line.split(",")
        print(a)
        vid_id = a[0]  # first element is video id
        toa = str(a[1])  # 2nd element is toa
        risk_object = a[2:-1]  # risk objects are from 3rd
        last_risk = a[-1].split('\n')[0]  # excluding the new line \n
        risk_object.append(last_risk)
        print(risk_object)
        risk_objects = [int(num) for num in risk_object]  # converting to int

        vid_ids.append(vid_id)  # Unnecessary
        # deepsort generated text files directory (video tracking information stored in this text file)
        vid_text = track_text + vid_id+'.txt'
        dummy_list = []
        with open(vid_text, 'r') as vid_text_file:
            for line in vid_text_file.readlines():
                a = line.split(" ")
                b = a[:6]  # we only need first 6 elements from the text file
                c = [int(val) for val in b]  # converting int
                c[4] = c[2] + c[4]  # converting width (w) to coordinate y2
                c[5] = c[3] + c[5]  # converting height (h) to coordinate x2
                c[0] = c[0]-1  # frame number starts at 0
                dummy_list.append(c)
            vid_text_file.close()
        df = pd.DataFrame(np.array(dummy_list), columns=[
                          'frame', 'track_id', 'y1', 'x1', 'y2', 'x2'])
        unique, counts = np.unique(df['frame'].values, return_counts=True)
        df['object_num'] = None
        for i in range(len(unique)):
            df.loc[df['frame'] == unique[i], 'object_num'] = np.array(
                range(counts[i]))  # counting the number of objects in each frame
        df['label'] = 0
        for i in range(len(risk_objects)):
            # label 1= risky object. Assigning labels corresponding to risk object.
            df.loc[df['track_id'] == risk_objects[i], 'label'] = int(1)
        # a csv file will be stored corresponding to each video_id
        df.to_csv('csv_files/'+vid_id+'-'+toa + '.csv', index=False)


# features : 100 x 35 x 512
# detections : 100 x 34 x 6 :
        # detections: track_id x bbox  x labels
        # track_id : 1
        # bbox: 4 coordinates ((normalized to 224,224 dimension) (upper_left, lower right) (w->,h))
        # labels 1 (choice 0 : negative or 1: positive)
# vid : 1 (string)
