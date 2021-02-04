# Dataset Analysis

## Conversational

* ***DailyDialogue***   
    * **size**: 11,118/1,000/1,000 for train/validation/test
    * **features**: 
        * avg length: `~30-`
        * multi-turn dialogue
        * 10 topics in total: 1. `Ordinary Life`, 2. `School Life`, 3. `Culture & Education`, 4. `Attitude & Emotion`, 5. `Relationship`, 6. `Tourism` , 7. `Health`, 8. `Work`, 9. `Politics`, 10. `Finance`
        * daily & natural

## Non-Conversational

* ***Movie Review***
    * **size**: 25,000/25,000 for train/test
    * **features**
        * avg length: `~150+`
        * balanced labels ( positive : negative = 1 : 1 )
        * neat
        * subjective

* ***Twitter for Cornell***
    * **size**: 1000/1000 for pos/neg
    * **features**
        * spoken language
        * some are dialogues focusing on the posts, need to filter out
        * some sentences are in the form of `xx : xxxx...`


* ***Twitter for Stanford***
    * too much `#xx` and `@xx`