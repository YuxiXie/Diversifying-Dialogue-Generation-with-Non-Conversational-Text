# Diversifying-Dialogue-Generation-with-Non-Conversational-Text
Implementation for the paper [Diversifying Dialogue Generation with Non-Conversational Text](https://www.aclweb.org/anthology/2020.acl-main.634) on English

Adapt from 
```
@inproceedings{su-etal-2020-diversifying,
    title = "Diversifying Dialogue Generation with Non-Conversational Text",
    author = "Su, Hui  and
      Shen, Xiaoyu  and
      Zhao, Sanqiang  and
      Xiao, Zhou  and
      Hu, Pengwei  and
      Zhong, Randy  and
      Niu, Cheng  and
      Zhou, Jie",
    booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.acl-main.634",
    doi = "10.18653/v1/2020.acl-main.634",
    pages = "7087--7097"
}
```

## Code Structure
```
|
|—— .gitignore
|—— README.md
|—— LICENSE
|—— scripts
|   |—— run_forward.sh
|   |—— get_forward_data.sh
|—— src
|   |—— forward
|   |—— backward
|   |__ topic-classifier
|__ data
	|—— data
	|	|__ utils.py
    |—— convert_to_classification_data_dailydialogue.py
    |—— convert_to_classification_data_empatheticdialogue.py
    |—— process_dailydialogue.py
    |—— process_ELI5.py
    |—— process_empatheticdialogue.py
    |—— process_twitter.py
    |—— process_wikihow.py
    |—— README.md
    |—— topic_stats.py
    |__ topic_stats_combine.py
```

---

## Datasets
### Conversational
* [DailyDialog](http://yanran.li/dailydialog)

* [EmpatheticDialogues](https://github.com/facebookresearch/EmpatheticDialogues)

### Non-conversational
* [Twitter](https://github.com/shaypal5/awesome-twitter-data)

* [WikiHow]()

* [ELI5](https://www.aclweb.org/anthology/P19-1346)

## Experiment Setup 
```
pip install -r requirements.txt
```

## Train

### Dialogue Generation
1. Initilization of forward & backward Seq2Seq models
    ```
    bash scripts/get_forward_data.sh
    bash scripts/run_forward.sh

    bash scripts/get_backward_data.sh
    bash scripts/run_backward.sh
    ```
    _PS_: After data processing by `get_forward/backward_data.sh`, the processed data is output in `.csv`-file with headers `prefix,input_text,target_text`

### Topic Classification


## Evaluate
