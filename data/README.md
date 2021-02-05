# Dataset Analysis

## Conversational

* ***DailyDialogue***   
    * **size**: 11,118/1,000/1,000 for train/validation/test
    * **features**: 
        * avg length: `~30-`
        * multi-turn dialogue
        * 10 topics in total: 1. `Ordinary Life`, 2. `School Life`, 3. `Culture & Education`, 4. `Attitude & Emotion`, 5. `Relationship`, 6. `Tourism` , 7. `Health`, 8. `Work`, 9. `Politics`, 10. `Finance`
        * daily & natural

* ***EmpatheticDialogue***
    * **size**: 19,533/2,770/2,547 for train/valid/test
    * **featrues**
        * multi-turn
        * with situation and label for each conversation

## Non-Conversational

* ***Movie Review***
    * **size**: 25,000/25,000 for train/test
    * **features**
        * avg length: `~150+`
        * balanced labels ( positive : negative = 1 : 1 )
        * neat
        * subjective

* ***Twitter for Stanford***
    * **size**: 1600000
    * **features**
        * avg original length: `~66.6` words 
            * (twitter has 280 character limit)
        * avg new length (>100 words removed): `~66.6` words
            * same as before
        * labels (negative : neutral : positive = 800k:0:800k)
            * no neutral labels
        * all @mentions and #hashtags are removed
    