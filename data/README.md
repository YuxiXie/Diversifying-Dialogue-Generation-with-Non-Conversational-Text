# Dataset Analysis

## Conversational

* ***DailyDialogue***   
    * **size**: 11,118/1,000/1,000 for train/validation/test
    * **features**: 
        * avg length: `~30-`
        * multi-turn dialogue
        * 10 topics in total: 1. `Ordinary Life`, 2. `School Life`, 3. `Culture & Education`, 4. `Attitude & Emotion`, 5. `Relationship`, 6. `Tourism` , 7. `Health`, 8. `Work`, 9. `Politics`, 10. `Finance`
        * daily & natural
        
    * format:
    ```
    [
      {
        "topic": "Work",
        "length": 2,
        "content": [
          {
            "text": "This position demands a higher writing ability , so please say something about your writing ability .",
            "act": "question",
            "emotion": ""
          },
          {
            "text": "Of course . I've loved writing since I was a very little boy . I won the first prize in a national composition contest when I was in middle school . After attending Nanjing University , I never give up writing . My works , such as Father's Tobacco Pipe , Open Air Cinema , The old City were respectively published China Youth Daily , Yangzi Evening News , and New Beijing . During the period of studying for my degrees of master and doctor , I paid more attention to developing my research ability and published several papers . The Impact of Internet in Chinese Political Participation , The Discipline of Remold , The Historical Direction of Chinese Administration Reform , Bribery Cases of Self governance in Chinese Villages are respectively published in Chinese Publish Administration , Beijing Due Xuebao , Theory and Society and Chinese Reform . I joined in Yangzi Evening News to work as a part-time journalist in 2006 . During this period , I've written a lot of comments , which improved my writing ability to a new level , I have full confidence in my writing ability , and I believe I can do the job well .",
            "act": "inform",
            "emotion": ""
          }
        ]
      },
      .
      .
      .
    ]
    ```
    * classification format:
    ```
    [
      {
        "text": "How do you like the pizza here ?",
        "topic": "Attitude & Emotion"
      },
      .
      .
      .
    ]
    ```

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

* ***ELI5***
    * **size**: 72w
    * **features**
        * QA dataset: labels contain `declarative` & `question`
        
