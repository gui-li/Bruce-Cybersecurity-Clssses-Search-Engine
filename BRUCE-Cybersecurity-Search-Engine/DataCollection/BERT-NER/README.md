# BERT + CRF NER

Use google BERT to do CoNLL-2003 NER !

[BERT-SQuAD](https://github.com/kamalkraj/BERT-SQuAD)


# Requirements

-  `python3`
- `pip3 install -r requirements.txt`

# Run

`python run_ner_crf.py --data_dir=course_data/ --bert_model=bert-base-cased --task_name=ner --output_dir=out_!x --max_seq_length=128 --do_train --num_train_epochs 5 --do_eval --warmup_proportion=0.4`


# Result

## BERT-BASE

### Test Data
```
               precision    recall  f1-score

     Textbook     0.9298    0.9636    0.9464
 Course title     0.9224    0.9145    0.9185
       Person     0.9904    0.9872    0.9888
Course number     0.9781    0.9824    0.9802
        Email     0.9286    1.0000    0.9630

  avg / total     0.9658    0.9727    0.9691
```
## Pretrained model download from [here](https://drive.google.com/file/d/1u3Bk96yeP01l_0TEt_HK4_fZwcy2W7-a/view?usp=sharing)

# Inference

```python
from bert import Ner

model = Ner("out_!x_course_crf/")

output = model.predict("CS231n: Computer Convolutional Neural Networks for Visual Recognition")

print(output)
'''
[
    {
        'word': 'CS231n',
        'tag': 'B-Course_number',
        'confidence': 0.9992440938949585
    },
    {
        'word': ':',
        'tag': 'O',
        'confidence': 0.9993892908096313
    },
    {
        'word': 'Computer',
        'tag': 'B-Course_title',
        'confidence': 0.9979397654533386
    },
    {
        'word': 'Convolutional',
        'tag': 'I-Course_title',
        'confidence': 0.9984481334686279
    },
    {
        'word': 'Neural',
        'tag': 'I-Course_title',
        'confidence': 0.9984042048454285
    },
    {
        'word': 'Networks',
        'tag': 'I-Course_title',
        'confidence': 0.998498797416687
    },
    {
        'word': 'for',
        'tag': 'I-Course_title',
        'confidence': 0.9980002045631409
    },
    {
        'word': 'Visual',
        'tag': 'I-Course_title',
        'confidence': 0.9966997504234314
    },
    {
        'word': 'Recognition',
        'tag': 'I-Course_title',
        'confidence': 0.9959718585014343
        }
]
'''
```

# Deploy REST-API
BERT NER model deployed as rest api
```bash
python api.py
```
API will be live at `0.0.0.0:8000` endpoint `predict`
#### cURL request
` curl -X POST http://0.0.0.0:8000/predict -H 'Content-Type: application/json' -d '{ "text": "Steve went to Paris" }'`

Output
```json
{
    "result": [
        {
            "confidence": 0.9981840252876282,
            "tag": "B-PER",
            "word": "Steve"
        },
        {
            "confidence": 0.9998939037322998,
            "tag": "O",
            "word": "went"
        },
        {
            "confidence": 0.999891996383667,
            "tag": "O",
            "word": "to"
        },
        {
            "confidence": 0.9991968274116516,
            "tag": "B-LOC",
            "word": "Paris"
        }
    ]
}
```


### Tensorflow version

- https://github.com/kyzhouhzau/BERT-NER

### Reference

- https://github.com/kamalkraj/BERT-NER/tree/experiment