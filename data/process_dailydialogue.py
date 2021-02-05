import sys
from tqdm import tqdm
from data.utils import json_dump


topics_dict = ['', 'Ordinary Life', 'School Life', 'Culture & Education',
          'Attitude & Emotion', 'Relationship', 'Tourism' , 'Health', 
          'Work', 'Politics', 'Finance']
acts_dict = ['', 'inform', 'question', 'directive', 'commissive']
emotions_dict = ['', 'anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise']


def get_text(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        data = f.read().strip().split('\n')
    return data


def get_samples(text_file, act_file, emotion_file, topics):
    text, acts, emotions = get_text(text_file), get_text(act_file), get_text(emotion_file)
    samples, avg_length = [], 0
    for txt, act, emotion in tqdm(zip(text, acts, emotions)):
        topic = topics_dict[topics[txt]]
        txt, act, emotion = txt.split(' __eou__ '), act.split(), emotion.split()
        
        content = []
        for t, a, e in zip(txt, act, emotion):
            utterance = {'text': t.rstrip('__eou__').strip(), 'act': acts_dict[int(a)], 'emotion': emotions_dict[int(e)]}
            content.append(utterance)
        sample = {'topic': topic, 'length': len(content), 'content': content}
        samples.append(sample)
    samples.sort(key=lambda x:x['length'])
    return samples
        

if __name__ == '__main__':
    overall_filename, topic_filename = sys.argv[1], sys.argv[2]
    topics_list = {o:int(t) for o, t in zip(get_text(overall_filename), get_text(topic_filename))}

    input_text, input_act, input_emotion = sys.argv[3], sys.argv[4], sys.argv[5]
    output_filename = sys.argv[6]

    samples = get_samples(input_text, input_act, input_emotion, topics_list)
    json_dump(samples, output_filename)