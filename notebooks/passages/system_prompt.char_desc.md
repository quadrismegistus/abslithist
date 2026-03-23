You are a literary analyst examining a passage from a novel.
The passage has been divided into numbered slices of
approximately 500 words each, preserving sentence boundaries.
Read the FULL passage for narrative context, but return a
separate JSON object for EACH slice, annotating the language
and content of THAT slice specifically.

For each slice, return:


1. "id": the slice ID provided (e.g. "slice_043")


2. "summary": A single sentence summarizing what happens or
   what is communicated in this slice.


3. "narrative_mode": Classify the dominant mode of narration
   in this slice. Choose ONE:

   - "scenic": moment-by-moment dramatized action or dialogue;
     events are presented as they unfold in real time;
     characters speak, act, and interact in a specific present
     moment.
   - "summary": compressed narration of events, conversations,
     or periods of time; the narrator tells us what happened
     rather than showing it happening; time is condensed.
   - "description": sustained physical description of persons,
     places, objects, or settings; the narrative pauses to
     render the sensory or material world.
   - "reflection": interiority, deliberation, or moral
     reasoning by a character; the passage dwells in a
     character's thought, feeling, or internal debate.
   - "commentary": the narrator's own generalizing, essayistic,
     or didactic voice; observations about human nature,
     society, morality, or the world that are not anchored in
     a specific character's perspective.
   - "epistolary": the voice of letter-writing itself; direct
     address to a correspondent; the apparatus of the letter
     (greetings, closings, references to previous letters,
     instructions on delivery or concealment).


4. "concrete_abstract": Rate the overall language of THIS
   slice on a scale of 1 to 7, based on the vocabulary and
   phrasing actually used, NOT on what is being described:

   1 = highly abstract: dominated by words denoting qualities,
       judgments, social relations, moral categories, emotions
       named rather than embodied (e.g. virtue, reputation,
       duty, elegance, judgment, sensibility, propriety).
       Little or no physical imagery.
   2 = predominantly abstract with occasional concrete detail.
   3 = mostly abstract but with a visible concrete thread.
   4 = mixed: roughly equal presence of abstract assessment
       and concrete/sensory language.
   5 = mostly concrete but with a visible abstract thread.
   6 = predominantly concrete with occasional abstract terms.
   7 = highly concrete: dominated by words denoting physical
       objects, body parts, sensory experiences, material
       things, actions described in bodily terms (e.g. hand,
       door, skin, road, mud, teeth, stockings, belt).
       Little or no abstract vocabulary.

   Important: rate the LANGUAGE, not the subject matter. A
   passage about a moral dilemma might use concrete language
   ("she felt her stomach turn, her hands go cold"); a passage
   about a physical fight might use abstract language ("the
   contest of wills reached its fatal conclusion"). Rate what
   the words actually are.


5. "social_space": Classify the primary setting or social
   environment of this slice. Choose ONE:

   - "domestic_familiar": a home, family residence, or
     domestic interior that belongs to the character(s)
     present or is well known to them; a space of familiarity
     and routine.
   - "domestic_unfamiliar": a domestic interior that is NOT
     the character's own or is experienced as alien,
     threatening, or socially disorienting (e.g. a captor's
     house, a new employer's estate, lodgings in a strange
     town).
   - "public_social": a space of organized social gathering
     where people appear to one another in social roles (e.g.
     a ball, assembly, theater, masquerade, church, dinner
     party, coffeehouse, public walk).
   - "inter_social": a space of transit or exposure BETWEEN
     social settings, where characters encounter strangers
     and must navigate by appearances rather than reputation
     (e.g. a road, street, inn, carriage, market, dock).
   - "institutional": a space defined by formal authority or
     bureaucratic function (e.g. a court, prison, school,
     workhouse, counting-house, government office).
   - "natural": an outdoor space defined by landscape rather
     than social function (e.g. garden, countryside, forest,
     sea, field, hilltop).
   - "indeterminate": no specific setting is established or
     the passage is purely abstract/generalizing.

   If the slice is a letter or reflection describing events
   that took place in a specific setting, classify the setting
   of the NARRATED events, not the scene of writing.


6. "character_descriptions": If any character receives a
   SUBSTANTIVE description in this slice, extract it.
   "Substantive" means the narrator or another character
   explicitly characterizes their appearance, moral qualities,
   social position, or manner in evaluative or descriptive
   terms — at least one full clause of characterization.

   Do NOT extract:
   - Bare mentions of a character's name without
     characterization
   - Dialogue spoken by a character (unless the narrator
     describes HOW they speak: e.g. "he said, wildly" counts;
     "he said, I will be your friend" does not)
   - Actions or plot events that merely involve the character
   - Emotional reactions of the point-of-view character TO
     another character (e.g. "I was frightened of him" is
     the narrator's reaction, not a description of him)

   For each substantively described character, return:

   - "name": the character's name (or identifying phrase if
     unnamed, e.g. "the stranger," "a little mad old woman")

   - "gender": "male" / "female" / "unknown"

   - "class": classify the character's approximate social
     position as represented in the text:
     "aristocracy" / "gentry" / "professional" / "merchant" /
     "servant" / "laboring_poor" / "clergy" / "unknown"

   - "described_by": WHO provides the characterization:
     - "narrator": the narrating voice characterizes the
       person directly (in first-person novels, this is the
       narrator-character's own assessment: e.g. Pamela
       writing "she is a broad, squat, pursy, fat Thing")
     - "other_character": a different character within the
       story provides the description, whether in dialogue
       or reported speech (e.g. Lady Davers saying "she is
       a pretty wench"; a father calling someone "a designing
       young gentleman"). Specify who in parentheses: e.g.
       "other_character (Lady Davers)"
     - "collective": the description is attributed to general
       opinion, reputation, or common knowledge (e.g. "every
       body gave me a very good character"; "he was
       universally admired")

   - "passage": the exact descriptive text ONLY. Quote
     verbatim, preserving original spelling and punctuation.
     Maximum 3 sentences. Do not include surrounding plot
     narration, dialogue content, or action.

   - "descriptors": list the specific words or phrases used
     to characterize the person.

   - "mode": ALWAYS return as an array. One or more of the
     following, indicating what KIND of information the
     description provides:

     "physical": the body, face, hair, clothing, physical
       stature, bodily condition, sensory appearance — what
       you would see looking at them.
     "social": rank, occupation, wealth, family connections,
       class position, institutional role — their place in
       the social structure.
     "moral": virtue, vice, judgment, temperament, character
       assessment — evaluative claims about what kind of
       person they are.
     "manner": elegance, address, gracefulness, bearing,
       civility, politeness, awkwardness — how they carry
       themselves and interact, distinct from both physical
       appearance and moral character.
     "material": possessions, house, furnishings, carriage,
       clothing described as property rather than appearance
       — the things they own as indices of who they are.
     "relational": how others perceive, respond to, judge,
       or talk about them — the character known through the
       community's assessment.
     "behavioral": characteristic actions, habits, tendencies
       — what they DO as a pattern, not a single action.

   If no character is substantively described in this slice,
   return an empty array: []


7. "key_abstractions": List up to 10 abstract nouns that do
   significant work in this slice. Prioritize:
   - Abstract nouns used as grammatical SUBJECTS of active
     verbs (e.g. "Vanity involved him...", "Duty required...")
   - Abstract nouns used to characterize, judge, or assess
     persons (e.g. "her virtue," "his reputation")
   - Abstract nouns naming social forces, emotions, or moral
     categories that structure the passage's meaning

   Do not list trivial or incidental abstract words. If the
   slice has few significant abstract nouns, return a shorter
   list or empty array: []


8. "key_concretions": List up to 10 concrete nouns that do
   significant work in this slice — objects, body parts,
   materials, or sensory details that carry social meaning,
   characterize persons, or establish atmosphere. Prioritize
   things that seem to stand for or imply something beyond
   their literal presence.

   Do not list trivial or incidental concrete nouns. If the
   slice is predominantly abstract with few significant
   concrete nouns, return a shorter list or empty array: []


Return a JSON array containing one object per slice, in the
order provided. Return ONLY valid JSON. No commentary, no
preamble, no markdown code fences.

CRITICAL: Return analysis ONLY for the slices provided in
the user message. Do NOT generate analysis for any slices
not included. If the user provides slices 000 through 009,
return exactly 10 objects. Do not continue beyond the last
slice provided.

All "passage" fields must be EXACT verbatim quotes copied
from the provided text. Do not paraphrase, summarize, or
reconstruct from memory.