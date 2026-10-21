# 0004 sample excerpts (PE vs NoPE, same sampling seed)

First ~400 characters of sample 0 (seed 42, prompt " ", temperature 0.8, top_k 40) from each checkpoint.
Both arms get the same random stream, so differences come from the model alone.

## early (iter 2000)

**PE (with positional embedding):**

```
after a few die aged on the other, and composed his on the forest, as they would be a long secroom and actually resolved that they were consciously becommenced, and that were long, when they returned to the narrative passengers.
Our sailor-march of the river fell on the spot, and not established about the reiners of the manner than a car. They had helped their treasury as they looked them at an of
```

**NoPE (no positional embedding):**

```
after our schools. Then stop worked away. Boston assented to himself it flattered above, but those scenes came his own during the reason of St. disposed at the accomplished sultent, he died not send those that had had passed and put in the minutes to mark, in the gentlement where not establish of battle, and were obliged and listened to the landing to the ship of the propotions of the deadly and f
```

## middle (iter 6000)

**PE (with positional embedding):**

```
after a few days, the time how a wide hand is of a source of mine. This flattery is a strange body of the cattle who can reach a creek of papers in the waters of the streets, like the animal of the rocky wind hill. The stream is pursued with ruins to mark, and presents its name when the water was battered.
Before the whole man sits at an end, a score of papers are repeated all looked the way that 
```

**NoPE (no positional embedding):**

```
after a few days, Lad took home with him. His own senses was made only sufficiently fortunate to go away. After I wondered by Mrs Fayther than she had remembered him, and had long been dispensed by the best a great property of days with the fruitless man and fruit empty spoken than to him. But they had not a week or two more fruitfully received in her friend. The husbands they looked down at the f
```

## late (iter 10000, final)

**PE (with positional embedding):**

```
after a few days' exile was on a plank compartment to the fire. He also filled the bay the second time.
Starting from the right street on the way, he would probably have been looking through on the road for a day. He had a pickaxe for his own base-tramps and he would show him he would have given up the wounded road. A few hours later the commanding officers arrived at the table with a dead clangor
```

**NoPE (no positional embedding):**

```
after a few days, Leila said would have been a member of the party alive. "It is not your family," she said.
"I can stand her family to the party," said the start. "It is not likely that, sir. I have no difficulty in passing and putting the little town a blind paw enough for me to pay her."
"She may be worried with me for her. I am afraid she is coming to share there is M. Flossie and my writing f
```
