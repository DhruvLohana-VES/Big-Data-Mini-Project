# Human Gold Annotation Guidelines
## Code-Mixed Hinglish and Marathi Social Text Analytics

### 1. Scope and Objective
These guidelines provide rigorous, standardized instructions for human annotators labelling multilingual, code-mixed social media posts (Hindi-English Hinglish, Marathi in Devanagari script, Romanized Marathi, and Mixed scripts).

### 2. Classification Tasks

#### A. Sentiment Classification
Annotate the primary polarity expressed by the author toward the target topic:
- **Positive (`positive`)**: Expresses satisfaction, admiration, praise, excitement, agreement, humor, or gratitude (e.g., *"Bhai ye movie kya mast banayi hai superb direction"*, *"खूपच सुंदर चित्रपट आहे"*).
- **Negative (`negative`)**: Expresses dissatisfaction, anger, frustration, disappointment, criticism, insult, or sarcastic mock praise (e.g., *"Arre yaar service bohot ghatiya hai"*, *"Wah bhai flight 3 ghante late, shabaash!"*).
- **Neutral (`neutral`)**: Factual updates, objective news, schedules, questions, weather, or statements devoid of emotional stance (e.g., *"Today cricket match is scheduled at 7 PM IST"*, *"दुपारी ३ वाजता पत्रकार परिषद होणार आहे"*).

#### B. Toxicity Classification
- **Toxic (`toxic`)**: Contains abusive language, hate speech, severe insults, aggressive threats, personal attacks, or vulgar profanity (e.g., *"Tum jaise kutto ko road pe pitna chahiye bloody scumbags"*, *"Aukaat madhe raha"*).
- **Non-Toxic (`non-toxic`)**: Constructive criticism, polite complaints, positive praise, humor, sarcasm without abusive slurs, or neutral discussion.

#### C. Linguistic & Pragmatic Attributes
- **Sarcasm (`sarcasm = true/false`)**: Detects irony where the literal lexical meaning contradicts the pragmatic negative intent (e.g., *"Wah kya governance hai, road pe gaddhe hain ya gaddho mein road!"*).
- **Script Mixing (`script_mixing = true/false`)**: True if the post contains simultaneous Devanagari and Latin (Roman) alphabet tokens in the same sentence (e.g., *"Superb काम केलं आहे @team"*).

### 3. Edge Cases & Code-Mixed Nuances
- **Phonetic Transliteration**: Variations like *kya / kyaaa*, *bohot / bahut*, *khup / khoop* should be interpreted in context.
- **Polysemous Slang**: Slang words like *bhai*, *bhava*, *raada*, *kadak* take sentiment depending on surrounding modifiers.
- **Sarcastic Praise**: If words look positive (*"shabaash"*, *"great job"*) but the post clearly points out a failure, mark Sentiment as `negative` and Sarcasm as `true`.

### 4. Quality Control & Agreement
- Each text is annotated independently by 3 annotators.
- Multi-annotator agreement is quantified via **Cohen's Kappa ($\kappa$)** (pairwise) and **Krippendorff's Alpha ($\alpha$)** (global multi-rater reliability).
- Gold master labels are established by majority consensus with expert adjudication on ties.
