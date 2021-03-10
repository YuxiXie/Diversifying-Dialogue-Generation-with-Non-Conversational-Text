# Dataset Analysis

## Conversational

* ***DailyDialogue***   
    * **size**: 11,118/1,000/1,000 for train/validation/test
    * **features**: 
        * avg length: `~30-`
        * multi-turn dialogue
        * 10 topics in total: 
          1. `Relationship`: 32807
          2. `Ordinary Life`: 29256
          3. `Work`:14802
          4. `Tourism`: 8504
          5. `School Life`: 4556
          6. `Finance`: 4248
          7. `Attitude & Emotion`: 4067
          8. `Health`: 2632
          9. `Politics`: 1583
          10. `Culture & Education`: 524
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
    * format

    ```
    [
      {
        "context": "devastated",
        "length": 2,
        "prompt": "When my mom got really ill and was in the hospital I wasn't able to visit her for about a week. It just also happened to be the week that she died. I was broken. I couldn't be there when she needed me the most. ",
        "content": [
          "The worst thing happened though. I wasn't able to see her for about a week because of not being allowed a bunch of time off",
          "The company you work for is horrible. They should have gave you that time off because it was a family emergency."
        ]
      },
      {
        "context": "confident",
        "length": 2,
        "prompt": "I felt really good about myself when i got a job at a local firm. I felt a sudden boost in my self worth",
        "content": [
          "i felt really good about myself when i got a new job at a reputed firm",
          "Wow. Congratulations! Is your new job near home?"
        ]
      },
      .
      .
      .
    ]
    ```

    * classification format

    ```
    [
      {
        "text": "The worst thing happened though. I wasn't able to see her for about a week because of not being allowed a bunch of time off",
        "topic": "devastated"
      },
      {
        "text": "The company you work for is horrible. They should have gave you that time off because it was a family emergency.",
        "topic": "devastated"
      },
     .
     .
     .
    ]
    ```

    

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
    * **size**: 720,000
    * **features**
        * QA dataset: labels contain `declarative` & `question`
    
* ***Trump Tweets***
    * **size**: 60,252
    * **features**
      * all @mentions, #hashtags and http addresses are removed
* ***Inaugural***
    * **size**: 5,153
    * **features**
      * neat and  complete speeches, no further process needed
* ***Medical-NLP***
    * **size**: 4,993
    * **features**
      * all corrupted characters are removed
* ***Stanford Sentiment Treebank***
    * **size**: 11,855
    * **features**
      * nice and easy reading sentences
      * problems with punctuation marks (unnecessary spaces, incorrect quotation marks) are resolved
* ***Sentiment140***
    * **size**: 1,048,523 (top 10,000 chosen)
    * **features**
      * all @mentions, #hashtags, http addresses and html marks are removed

