# Fantasy Sports NLG
## About
A system for hierarchical natural language generation in the domain of fantasy sports news.
Automatically generates templates from a source corpus and uses data-to-text strategies to generate summaries of a player’s performance in a game.

We gathered a parallel corpus of data and text about a particular player and game to generate these templates. We then generated text hierarchically by performing document planning, microplanning, and surface realization.

Templates improve microplanning and surface realization by maintaining syntactic and semantic structure from the source corpus. Where text aligns with data in our table we can create slots in the templates for inserting future data and generating new text. The downside of templates is that they lack variation so we break the templates from the source text into chunks which we can recombine creating a balance between these aspects.

## Dataset
We scrapped rotoworld.com for news updates about players and footballdb.com for player statistics. This allows us to construct a parallel corpus for learning data-to-text generation. 
We annotated one week of news updates as our gold corpus to compare our approaches against. We ended up with 2,118 training samples and 160 labeled test samples.

## Methods
For the document planning, microplanning, and surface realization tasks we create features following guidelines in Angeli et al. (2010) Section 3.

Several approaches to template generation were explored including: 
- MsApriori Frequent-Chunksets
- Combinatorics
- Recurrent Neural Networks
- Logistic Regression
- Naive Bayes

## Execution


## View presentation
https://bit.ly/2Eq9VuP
