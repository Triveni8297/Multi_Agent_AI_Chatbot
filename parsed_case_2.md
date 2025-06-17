See discussions, stats, and author profiles for this publication at: https://www.researchgate.net/publication/370653602
Generative AI
Preprint · May 2023
CITATIONS                                                                                       READS
0                                                                                               27,654
4 authors, including:
          Jochen Hartmann                                                                                 Patrick Zschech
          Technische Universität München                                                                  University of Leipzig
          27 PUBLICATIONS 648 CITATIONS                                                                   67 PUBLICATIONS  1,408 CITATIONS
            SEE PROFILE                                                                                      SEE PROFILE
 All content following this page was uploaded by Patrick Zschech on 05 October 2023.
 The user has requested enhancement of the downloaded file.

Bus Inf Syst Eng
https://doi.org/10.1007/s12599-023-00834-7
 C A T C H WO R D
Generative AI
Stefan Feuerriegel    • Jochen Hartmann     • Christian Janiesch   •
Patrick Zschech
Received: 29 April 2023 / Accepted: 7 August 2023
Ó The Author(s) 2023
Keywords    Generative AI ℂ Artiﬁcial intelligence ℂ            The term  generative AI refers to computational tech-
Decision support ℂ Content creation ℂ Information systems      niques that are capable of generating  seemingly  new,
                                                               meaningful content such as text, images, or audio from
                                                               training data. The widespread diffusion of this technology
1 Introduction                                                 with examples such as Dall-E 2, GPT-4, and Copilot is
                                                               currently revolutionizing the way we work and communi-
Tom Freston is credited with saying ‘‘Innovation is taking     cate with each other. Generative AI systems can not only
two things that exist and putting them together in a new       be used for artistic purposes to create new text mimicking
way’’. For a long time in history, it has been the prevailing  writers or new images mimicking illustrators, but they can
assumption that artistic, creative tasks such as writing       and will assist humans as intelligent question-answering
poems, creating software, designing fashion, and compos-       systems. Here, applications include information technology
ing  songs  could only  be  performed  by  humans.   This      (IT) help desks where generative AI supports transitional
assumption has changed drastically with recent advances in     knowledge work tasks and mundane needs such as cooking
artiﬁcial intelligence (AI) that can generate new content in   recipes and medical advice. Industry reports suggest that
ways that cannot be distinguished anymore from human           generative AI could raise global gross domestic product
craftsmanship.                                                 (GDP) by 7% and replace 300 million jobs of knowledge
                                                               workers (Goldman Sachs 2023). Undoubtedly, this has
                                                               drastic implications not only for the Business & Informa-
Accepted after one revision by Susanne Strahringer.            tion Systems Engineering (BISE) community, where we
S. Feuerriegel (&)                                             will face revolutionary opportunities, but also challenges
LMU Munich and Munich Center for Machine Learning,             and risks that we need to tackle and manage to steer the
Geschwister-Scholl-Platz 1, 80539 Munich, Germany              technology and its use in a responsible and sustainable
e-mail: feuerriegel@lmu.de                                     direction.
J. Hartmann                                                     In this Catchword article, we provide a conceptualiza-
Technical University of Munich, TUM School of Management,      tion of generative AI as an entity in socio-technical systems
Arcisstr. 21, 80333 Munich, Germany                            and provide examples of models, systems, and applica-
e-mail: jochen.hartmann@tum.de                                 tions. Based on that, we introduce limitations of current
C. Janiesch                                                    generative AI and provide an agenda for BISE research.
TU Dortmund University, Otto-Hahn-Str. 12, 44319 Dortmund,     Previous papers discuss generative AI  around  speciﬁc
Germany                                                        methods such as language models (e.g., Teubner et al.
                                                               2023; Dwivedi et al. 2023;  ¨bel
e-mail: christian.janiesch@tu-dortmund.de                                               Scho    et al. 2023) or speciﬁc
P. Zschech                                                     applications such as marketing (e.g., Peres et al. 2023),
FAU           ¨rnberg,                     ¨rnberg,            innovation management  (Burger  et al. 2023), scholarly
     Erlangen-Nu      Lange Gasse 20, 90403 Nu                 research (e.g., Susarla et al. 2023; Davison et al. 2023),
Germany
e-mail: patrick.zschech@fau.de                                 and education (e.g., Kasneci et al. 2023; Gimpel et al.
Published online: 12 September 2023                                                                         1 3

2023). Different from these works, we focus on generative
AI in the context of information systems, and, to this end,
we discuss several opportunities and challenges that are
unique to the BISE community and make suggestions for
impactful directions for BISE research.

2 Conceptualization
2.1 Mathematical Principles of Generative AI
Generative AI is primarily based on generative modeling,
which has distinctive mathematical differences from dis-
criminative modeling (Ng and Jordan 2001) often used in
data-driven decision support. In general, discriminative
modeling tries to separate data points    X  into different
classes Y  by learning decision boundaries between them
(e.g., in classiﬁcation tasks with Y 2 f0; 1g). In contrast to
that, generative modeling aims to infer some actual data
distribution. Examples can be the joint probability distri-
bution P(X, Y) of both the inputs and the outputs or P(Y),
but  where  Y  is typically from  some   high-dimensional
space. By doing so, a generative model offers the ability to
produce new synthetic samples (e.g., generate new obser-
vation-target-pairs (X, Y) or new observations X   given a
target value Y) (Bishop 2006).
   Building upon the above, a generative AI model refers to
 generative modeling that is instantiated with a machine
 learning architecture (e.g., a deep neural network) and,
 therefore, can create new data samples based on learned
patterns.1 Further, a generative AI system encompasses the
entire infrastructure, including the model, data processing,
and user interface components. The model serves as the
core component of the system, which facilitates interaction
and application within a broader context. Lastly, generative
AI applications refer to the practical use cases and imple-
mentations of these systems, such as search engine opti-
mization (SEO) content generation or code generation that
solve real-world problems and drive innovation across
various  domains.  Figure 1  shows   a systematization  of
generative AI across selected data modalities (e.g., text,
image, and audio) and the model-, system-, and application-
level perspectives, which we detail in the following section.

1 It should be noted, however, that advanced generative AI models
are often not based on a single modeling principle or learning
mechanism, but combine different approaches. For example, language
models from the GPT family ﬁrst apply a generative pre-training
stage to capture the distribution of language data using a language
modeling objective, while downstream systems typically then apply a
discriminative ﬁne-tuning stage to adapt the model parameters to
speciﬁc tasks (e.g., document classiﬁcation, question answering).
Similarly, ChatGPT combines techniques from generative modeling
together with discriminatory modeling and reinforcement learning
(see Fig. 2).
 1 3
      S. Feuerriegel et al.: Generative AI, Bus Inf Syst Eng
Note that the modalities in Fig. 1 are neither complete
nor entirely distinctive and can be detailed further. In
addition, many unique use cases such as, for example,
modeling functional properties of proteins (Unsal et al.
2022) can be represented in another modality such as text.
2.2 A Model-, System-, and Application-Level View
of Generative AI
2.2.1 Model-Level View
A generative AI model is a type of machine learning
architecture that uses AI algorithms to create novel data
instances, drawing upon the patterns and relationships
observed in the training data. A generative AI model is of
critically central yet incomplete nature, as it requires fur-
ther ﬁne-tuning to speciﬁc tasks through systems and
applications.
Deep neural networks are particularly well suited for the
purpose of data generation, especially as deep neural net-
works can be designed using different architectures to
model different data types (Janiesch et al. 2021; Kraus
et al. 2020), for example, sequential data such as human
language or spatial data such as images. Table 1 presents
an overview of the underlying concepts and model archi-
tectures that are common in the context of generative AI,
such as diffusion probabilistic models for text-to-image
generation or the transformer architecture and (large) lan-
guage models (LLMs) for text generation. GPT (short for
generative pre-trained transformer), for example, repre-
sents a popular family of LLMs, used for text generation,
for instance, in the conversational agent ChatGPT.
Large generative AI models that can model output in
and across speciﬁc domains or speciﬁc data types in a
comprehensive and versatile manner are oftentimes also
called foundation models (Bommasani et al. 2021). Due to
their size, they exhibit two key properties: emergence,
meaning the behavior is oftentimes implicitly induced
rather than explicitly constructed (e.g., GPT models can
create calendar entries in the .ical format even though such
models were not explicitly trained to do so), and homog-
enization, where a wide range of systems and applications
can now be powered by a single, consolidated model (e.g.,
Copilot can generate source code across a wide range of
programming languages).
Figure 1 presents an overview of generative AI models
along different, selected data modalities, which are pre-
trained on massive amounts of data. Note that we structure

the models in Fig. 1 by their output modality such as X-to-
text or X-to-image. For example, GPT-4 as the most recent
generative AI model underlying OpenAI’s popular con-
versational agent ChatGPT (OpenAI 2023a) accepts both

image and text inputs to generate text outputs. Similarly,

S. Feuerriegel et al.: Generative AI, Bus Inf Syst Eng
Fig. 1 A model-, system-, and
application-level view on
generative AI
Midjourney accepts both modalities to generate images. To
this end, generative AI models can also be grouped into
unimodal and multimodal models. Unimodal models take
instructions from the same input type as their output (e.g.,
text). On the other hand, multimodal models can take input
from  different sources  and generate  output  in various
forms. Multimodal models exist across a variety of data
modalities, for  example   for  text, image,  and  audio.
Prominent examples include Stable Diffusion (Rombach
et al. 2022)   for  text-to-image  generation, MusicLM
(Agostinelli  et al. 2023) for  text-to-music generation,
Codex (Chen et al. 2021) and AlphaCode (Li et al. 2022)
for text-to-code generation, and as mentioned above GPT-4
for  image-to-text  as  well  as  text-to-text generation
(OpenAI 2023a).
   The underlying training procedures vary greatly across
different generative AI models (see Fig. 2). For example,
generative  adversarial  networks   (GANs)   are  trained
through  two   competing   objectives (Goodfellow   et al.
2014), where one is to create new synthetic samples while
the other tries to detect synthetic samples from the actual
training samples,  so  that the distribution of synthetic
samples  is eventually  close to  the distribution of the
training samples. Differently, systems such as ChatGPT-
based conversational models use reinforcement learning
from human feedback (RLHF). RLHF as used by ChatGPT
proceeds in three steps to ﬁrst create demonstration data for
prompts, then to have users rank the quality of different
outputs for a prompt, and ﬁnally to learn a policy that
generates desirable output via reinforcement learning so
that the output would score well during ranking (Ziegler
et al. 2019).






2.2.2 System-Level View

Any system consists of a number of elements that are
interconnected and interact with each other. For generative
AI systems, this comprises not only the aforementioned
generative AI model but also the underlying infrastructure,
user-facing components, and their modality as well as the
corresponding data processing (e.g., for prompts).  An
example would be the integration of deep learning models,
like Codex (Chen et al. 2021), into a more interactive and
comprehensive system, like GitHub Copilot, which allows
its users to code more efﬁciently. Similarly, Midjourney’s
image generation system builds on an undisclosed X-to-
image generation model that users can interact with to
generate images using Discord bots. Thus, generative AI
systems embed the functionality of the underlying mathe-
matical model to provide an interface for user interaction.
This  step augments   the  model-speciﬁc   capabilities,
enhancing its practicability and usability across real-world
use cases.
 Core concerns when embedding deep learning models in
generative AI systems generally are scalability (e.g., dis-
tributed computing resources), deployment (e.g., in various
environments and for different devices), and usability (e.g.,
a user-friendly interface and intent recognition). As pre-
trained open-source alternatives to closed-source, propri-
etary models continue to be released, making these models
available to their users (be it companies or individuals)
becomes increasingly important. For both open-source and
closed-source models, unexpected deterioration of model
performance over time highlights the need for continuous
model monitoring (Chen et al. 2023). Although powerful
text-generating models existed before the release of the
                                             1 3

                                                                                         S. Feuerriegel et al.: Generative AI, Bus Inf Syst Eng
Table 1  Glossary of key concepts in generative AI
Concept                              Description
Diffusion probabilistic models       Diffusion probability models are a class of latent variable models that are common for various tasks such as
                                     image generation (Ho et al. 2020). Formally, diffusion probability models capture the image data by
                                     modeling the way data points diffuse through a latent space, which is inspired by statistical physics.
                                     Speciﬁcally, they typically use Markov chains trained with variational inference and then reverse the
                                     diffusion process to generate a natural image. A notable variant is Stable Diffusion (Rombach et al. 2022).
                                     Diffusion probability models are also used in commercial systems such as DALL-E and Midjourney.
Generative adversarial network       A GAN is a class of neural network architecture with a custom, adversarial learning objective (Goodfellow
                                     et al. 2014). A GAN consists of two neural networks that contest with each other in the form of a zero-sum
                                     game, so that samples from a speciﬁc distribution can be generated. Formally, the ﬁrst network G is called
                                     the generator, which generates candidate samples. The second network D is called the discriminator, which
                                     evaluates how likely the candidate samples come from a desired distribution. Thanks to the adversarial
                                     learning objective, the generator learns to map from a latent space to a data distribution of interest, while
                                     the discriminator distinguishes candidates produced by the generator from the true data distribution (see
                                     Fig. 2).
(Large) language model               A (large) language model (LLM) refers to neural networks for modeling and generating text data that
                                     typically combine three characteristics. First, the language model uses a large-scale, sequential neural
                                     network (e.g., transformer with an attention mechanism). Second, the neural network is pre-trained through
                                     self-supervision in which auxiliary tasks are designed to learn a representation of natural language without
                                     risk of overﬁtting (e.g., next-word prediction). Third, the pre-training makes use of large-scale datasets of
                                     text (e.g., Wikipedia, or even multi-language datasets). Eventually, the language model may be ﬁne-tuned
                                     by practitioners with custom datasets for speciﬁc tasks (e.g., question answering, natural language
                                     generation). Recently, language models have evolved into so-called LLMs, which combine billions of
                                     parameters. Prominent examples of massive LLMs are BERT (Devlin et al. 2018) and GPT-3 (Brown et al.
                                     2020) with  340 million and  175 billion parameters, respectively.
Reinforcement learning from          RLHF learns sequential tasks (e.g., chat dialogues) from human feedback. Different from traditional
human feedback                       reinforcement learning, RLHF directly trains a so-called reward model from human feedback and then uses
                                     the model as a reward function to optimize the policy, which is optimized through data-efﬁcient and robust
                                     algorithms (Ziegler et al. 2019). RLHF is used in conversational systems such as ChatGPT (OpenAI 2022)
                                     for generating chat messages, such that new answers accommodate the previous chat dialogue and ensure
                                     that the answers are in alignment with predeﬁned human preferences (e.g., length, style, appropriateness)
Prompt learning                      Prompt learning is a method for LLMs that uses the knowledge stored in language models for downstream
                                     tasks (Liu et al. 2023). In general, prompt learning does not require any ﬁne-tuning of the language model,
                                     which makes it efﬁcient and ﬂexible. A prompt is a speciﬁc input to a language model (e.g., ‘‘The movie
                                     was superb. Sentiment: ‘‘) and then the most probable output s 2 f‘‘positive’’; ‘‘negative’’g instead of the
                                     space is picked. Recent advances allow for more complex data-driven prompt engineering, such as tuning
                                     prompts via reinforcement learning (Liu et al. 2023).
seq2seq                              The term sequence-to-sequence (seq2seq) refers to machine learning approaches where an input sequence is
                                     mapped onto an output sequence (Sutskever et al. 2014). An example is machine learning-based translation
                                     between different languages. Such seq2seq approaches consist of two main components: An encoder turns
                                     each element in a sequence (e.g., each word in a text) into a corresponding hidden vector containing the
                                     element and its context. The decoder reverses the process, turning the vector into an output element (e.g., a
                                     word from the new language) while considering the previous output to model dependencies in language.
                                     The idea of seq2seq models has been extended to allow for multi-modal mappings such as text-to-image or
                                     text-to-speech mappings.
Transformer                          A transformer is a deep learning architecture (Vaswani et al. 2017) that adopts the mechanism of self-
                                     attention which differentially weights the importance of each part of the input data. Like recurrent neural
                                     networks (RNNs), transformers are designed to process sequential input data, such as natural language, with
                                     applications for tasks such as translation and text summarization. However, unlike RNNs, transformers
                                     process the entire input all at once. The attention mechanism provides context for any position in the input
                                     sequence. Eventually, the output of a transformer (or an RNN in general) is a document embedding, which
                                     presents a lower-dimensional representation of text (or other input) sequences where similar texts are
                                     located in closer proximity which typically beneﬁts downstream tasks as this allows to capture semantics
                                     and meaning (Siebers et al. 2022).
Variational autoencoder              A variational autoencoder (VAE) is a type of neural network that is trained to learn a low-dimensional
                                     representation of the input data by encoding it into a compressed latent variable space and then
                                     reconstructing the original data from this compressed representation. VAEs differ from traditional
                                     autoencoders by using a probabilistic approach to the encoding and decoding process, which enables them
                                     to capture the underlying structure and variation in the data and generate new data samples from the learned
                                     latent space (Kingma and Welling 2013). This makes them useful for tasks such as anomaly detection and
                                     data compression but also image and text generation.
1 3

S. Feuerriegel et al.: Generative AI, Bus Inf Syst Eng
Table 1  continued
Concept                           Description
Zero-shot learning / few-shot     Zero-shot learning and few-shot learning refer to different paradigms of how machine learning deals with
learning                          the problem of data scarcity. Zero-shot learning is when a machine is taught how to learn a task from data
                                  without ever needing to access the data itself, while few-short learning refers to when there are only a few
                                  speciﬁc examples. Zero-shot learning and few-shot learning are often desirable in practice as they reduce
                                  the cost of setting up AI systems. LLMs are few-shot or zero-shot learners (Brown et al. 2020) as they just
                                  need a few samples to learn a task (e.g., predicting the sentiment of reviews), which makes LLMs highly
                                  ﬂexible as a general-purpose tool.
                                                                  2
Fig. 2 Examples of different training procedures for generative AI models. a Generative adversarial network (GAN). b Reinforcement learning
from human feedback (RLHF) as used in conversational generative AI models
ChatGPT system in November 2022, ChatGPT’s ease of                  in many generative AI models are that they were trained on
use also for non-expert users was a core contributing factor        historical data with speciﬁc cut-off date and thus do not
to its explosive worldwide adoption.                                store information beyond or that an information compres-
   Moreover, on the system level, multiple components of            sion takes place because of which generative AI models
a generative AI system can be integrated or connected to            may not remember everything that they saw during training
other   systems,  external  databases   with  domain-speciﬁc        (Chiang 2023). Both limitations can be mitigated by aug-
knowledge, or platforms. For example, common limitations            menting    the  model    with   functionality   for  real-time
                                                                                                                        1 3

      information retrieval, which can substantially enhance its
      accuracy and usefulness. Relatedly, in the context of text
       generation, online language modeling addresses the prob-
       lem of outdated models by continuously training them on
up-to-date  data.2 Thereby,   such  models   can  then  be
      knowledgeable of recent events that their static counter-
      parts would not be aware of due to their training cut-off
dates.

2.2.3 Application-Level View

      Generative AI applications are generative AI systems sit-
     uated in organizations to deliver value by solving dedicated
       business problems and addressing stakeholder needs. They
can  be  regarded  as human-task-technology    systems  or
       information systems that use generative AI technology to
        augment human capacities to accomplish speciﬁc tasks.
       This level of generative AI encompasses countless real-
       world use cases: These range from SEO content generation
      (Reisenbichler et al. 2022), over synthetic movie genera-
       tion (Metz 2023) and AI music generation (Garcia 2023),
         to natural language-based software development (Chen
et al. 2021).
   Generative  AI  applications  will give  rise to novel
        technology-enabled modes of work. The more users will
      familiarize themselves with these novel applications, the
       more they will trust or mistrust them as well as use or
     disuse them. Over time, applications will likely transition
       from mundane tasks such as writing standard letters and
     getting a dinner reservation to more sensitive tasks such as
      soliciting medical or legal advice. They will involve more
        consequential decisions, which may even involve moral
judgment      ¨gel
          (Kru     et al. 2023). This ever-increasing scope
     and pervasiveness of generative AI applications give rise to
        an imminent need not only to provide prescriptions and
    principles for trustworthy and reliable designs, but also for
     scrutinizing the effects on the user to calibrate qualities
such  as  trust appropriately. The   (continued) use  and
       adoption of such applications by end users and organiza-
tions  entails a number   of fundamental   socio-technical
       considerations to descry innovation potential and affor-
dances of generative AI artifacts.

2.3 A Socio-Technical View on Generative AI

       As technology advances, the deﬁnition and extent of what
     constitutes AI are continuously reﬁned, while the reference
       point of human intelligence stays comparatively constant

2 See https://github.com/huggingface/olm-datasets (accessed 25 Aug
2023) for a script that enables users to pull up-to-date data from the
  web for training online language models, for instance, from Common
Crawl and Wikipedia.
 1 3
  S. Feuerriegel et al.: Generative AI, Bus Inf Syst Eng
(Berente  et al. 2021). With  generative AI, we  are
approaching a further point of reﬁnement. In the past, the
capability of AI was mostly understood to be analytic,
suitable for decision-making tasks. Now, AI gains the
capability to perform generative tasks, suitable for content
creation. While the procedure of content creation to some
respect can still be considered analytic as it is inherently
probabilistic, its results can be creative or even artistic as
generative AI combines elements in novel ways. Further,
IT artifacts were considered passive as they were used
directly by humans. With the advent of agentic IT artifacts
(Baird and Maruping 2021) powered by LLMs (Park et al.
2023), this human agency primacy assumption needs to be
revisited and impacts how we devise the relation between
human and AI based on their potency. Eventually, this may
require AI capability models to structure, explain, guide,
and constrain the different abilities of AI systems and their
uses as AI applications.
Focusing on the interaction between humans and AI, so
far, for analytic AI, the concept of delegation has been
discussed to establish a hierarchy for decision-making
(Baird and Maruping 2021). With generative AI, a human
uses prompts to engage with an AI system to create con-
tent, and the AI then interprets the human’s intentions and
provides feedback to presuppose further prompts. At ﬁrst
glance, this seems to follow a delegation pattern as well.
Yet, the subsequent process does not, as the output of the
AI can be suggestive to the other and will inform their
further involvement directly or subconsciously. Thus, the
process of creation rather follows a co-creation pattern, that
is, the practice of collaborating in different roles to align
and offer diverse insights to guide a design process
(Ramaswamy and Ozcan 2018). Using the lens of agentic

AI artifacts, initiation is not limited to humans.
The abovementioned interactions also impact our cur-
rent understanding of hybrid intelligence as the integration
of humans and AI, leveraging the unique strengths of both.
Hybrid intelligence argues to address the limitations of
each intelligence type by combining human intuition, cre-
ativity, and empathy with the computational power, accu-
racy, and scalability of AI systems to achieve enhanced
decision-making and problem-solving capabilities (Deller-
mann et al. 2019). With generative AI and the AI’s capa-
bility to co-create, the understanding of what constitutes
this collective intelligence begins to shift. Hence, novel
human-AI interaction models and patterns may become
necessary to explain and guide the behavior of humans and
AI systems to enable effective and efﬁcient use in AI
applications on the one hand and, on the other hand, to
ensure envelopment of AI agency and reach (Asatiani et al.
2021).
On a theoretical level, this shift in human-computer or
rather human-AI  interaction  fuels  another important

S. Feuerriegel et al.: Generative AI, Bus Inf Syst Eng
observation: The theory of mind is an established theoret-
ical lens in psychology to describe the cognitive ability of
individuals to understand and predict the mental states,
emotions, and intentions of others (Carlson et al. 2013;
Baron-Cohen 1997; Gray et al. 2007). This skill is crucial
for social interactions, as it facilitates empathy and allows
for effective communication. Moreover, conferring a mind
to an AI system can substantially drive usage intensity
(Hartmann et al. 2023a). The development of a theory of
mind in humans is unconscious and evolves throughout an
individual’s life. The more natural AI systems become in
terms of their interface and output, the more a theory of
mind for human-computer interactions becomes necessary.
Research is already investigating how AI systems can
become theory-of-mind-aware to better understand their
human counterpart (Rabinowitz et al. 2018; C
                                            ¸ elikok et al.
2019). However, current AI systems hardly offer any cues
for interactions. Thus, humans are rather void of a theory to
explain their understanding of intelligent behavior by AI
systems, which becomes even more important in a co-
creation environment that does not follow a task delegation
pattern. A theory of the artiﬁcial mind that explains how
individuals perceive and assume the states and rationale of
AI systems to better collaborate with them may alleviate
some of these concerns.


3 Limitations of Current Generative AI

In the following, we discuss four salient boundaries of
generative AI that, we argue, are important limitations in
real-world applications. The following limitations are of
technical nature in that they refer to how current generative
AI models make inferences, and, hence, the limitations
arise at the model level. Because of this, it is likely that
limitations will persist in the long run, with system- and
application-level implications.
   Incorrect outputs. Generative AI models may produce
output with errors. This is owed to the underlying nature of
machine learning models relying on probabilistic algo-
rithms for making inferences. For example, generative AI
models generate the most probable response to a prompt,
not necessarily the correct response. As such, challenges
arise as,  by  now,  outputs  are  indistinguishable from
authentic  content and   may  present  misinformation  or
deceive users (Spitale et al. 2023). In LLMs, this problem
in emergent behavior is called hallucination (Ji et al. 2023),
which refers to mistakes in the generated text that are
semantically or syntactically plausible but are actually
nonsensical or incorrect. In other words, the generative AI
model produces content that is not based on any facts or
evidence, but rather on its own assumptions or biases.



Moreover, the output of generative AI, especially that of
LLMs, is typically not easily veriﬁable.
The  correctness  of generative  AI  models  is highly
dependent on the quality of training data and the according
learning process. Generative AI systems and applications
can implement correctness checks to inhibit certain out-
puts. Yet, due to the black-box nature of state-of-the-art AI
models (Rai 2020), the usage of such systems critically
hinges on users’ trust in reliable outputs. The closed source
of commercial off-the-shelf generative AI systems aggra-
vates this fact and prohibits further tuning and re-training
of the models. One solution for addressing the downstream
implications of incorrect outputs is to use generative AI to
produce explanations or references, which can then be
veriﬁed by users. However, such explanations are again
probabilistic and thus subject to errors; nevertheless, they

may help users in their judgment and decision-making
when to accept outputs of generative AI and when not.
Bias and fairness. Societal biases permeate everyday
human-generated content (Eskreis-Winkler and Fishbach
2022). The unbiasedness of vanilla generative AI is very
much dependent on the quality of training data and the
alignment process. Training deep  learning  models  on
biased data can amplify human   biases, replicate toxic
language, or perpetuate stereotypes of gender, sexual ori-
entation, political leaning, or religion (e.g., Caliskan et al.
2017; Hartmann et al. 2023b). Recent studies expose the
harmful biases embedded in multimodal generative AI
models such as CLIP (contrastive language-image pre-
training; Wolfe et al. 2022) and the CLIP-ﬁltered LAION
dataset (Birhane et al. 2021), which are core components
of generative AI models (e.g., Dall-E 2 or Stable Diffu-
sion). Human biases can also creep into the models in other
stages of the model engineering process. For instruction-
based language models, the RLHF process is an additional
source of bias (OpenAI 2023b). Careful coding guidelines
and quality checks can help address these risks.
Addressing   bias  and  thus  fairness in AI   receives
increasing attention in the academic literature (Dolata et al.
2022; Schramowski et al. 2022; Ferrara 2023; De-Arteaga
et al. 2022; Feuerriegel et al. 2020; von Zahn et al. 2022),
but remains an open and ongoing research question. For
example, the developers of Stable Diffusion ﬂag ‘‘probing
and understanding the limitations and biases of generative
models’’ as an important research area (Rombach et al.
2022). Some scholars even attest to models certain moral
self-correcting capabilities (Ganguli et al. 2023), which
may attenuate concerns of embedded biases and result in
more fairness. In addition, on the system and application
level, mitigation mechanisms  can  be implemented    to
address biases embedded in the deep learning models and
create more diverse outputs (e.g., updating the prompts
‘‘under the hood’’ as done by Dall-E 2 to increase the
                                             1 3

demographic diversity of the outputs). Yet, more research
is needed to get closer to the notion of fair AI.
   Copyright  violation. Generative AI models,  systems,
and applications may cause a violation of copyright laws
because they can produce outputs that resemble or even
copy existing works without permission or compensation to
the original creators (Smits and Borghuis 2022). Here, two
potential infringement risks are common. On the one hand,
generative AI may make illegal copies of a work, thus
violating the reproduction right of creators. Among others,
this may happen when a generative AI was trained on
original content that is protected by copyright but where
the  generative AI  produces   copies. Hence,  a  typical
implication is that the training data for building generative
AI systems must be free of copyrights. Crucially, copyright
violation may nevertheless still happen even when the
generative AI has never seen a copyrighted work before,
such as, for example, when it simply produces a trade-
marked   logo  similar to  that of  Adidas  but  without
ever having seen that logo before. On the other hand,
generative AI may prepare derivative works, thus violating
the transformation right of creators. To this end, legal
questions arise around the balance of originality and cre-
ativity in generative AI systems. Along these lines, legal
questions also  arise around  who  holds  the intellectual
property  for works  (including patents) produced   by a
generative AI.
   Environmental concerns. Lastly, there are substantial
environmental concerns from developing and using gen-
erative AI systems due to the fact that such systems are
typically built around large-scale neural networks, and,
therefore, their development and operation consume large
amounts   of electricity with immense   negative  carbon
footprint (Schwartz et al. 2020). For example, the carbon
emission for training a generative AI model such as GPT-3
was   estimated  to  have  produced   the  equivalent  of
552 t CO2 and thus amounts to the annual CO2   emissions
of several dozens of households (Khan 2021). Owing to
this, there are ongoing efforts in AI research to make the
development and deployment of AI algorithms more car-
bon-friendly, through more efﬁcient training algorithms,
through compressing the size of neural network architec-
tures, and through optimized hardware (Schwartz et al.
2020).


4 Implications and Future Directions for the BISE
  Community

In this section, we draw a number of implications and
future research directions which, on the one hand, are of
direct relevance to the BISE community as an application-
oriented, socio-technical research discipline and, on the
1 3
          S. Feuerriegel et al.: Generative AI, Bus Inf Syst Eng
other hand, offer numerous research opportunities, espe-
cially for BISE researchers due to their interdisciplinary
background. We organize our considerations according to
the individual departments of the  BISE   journal (see
Table 2 for an overview of exemplary research questions).

4.1 Business Process Management

Generative AI will have a strong impact on the ﬁeld of
Business Process Management (BPM) as it can assist in
automating  routine  tasks,  improving  customer  and
employee satisfaction, and revealing process innovation
opportunities (Beverungen et al. 2021), especially in cre-
ative processes (Haase and Hanel 2023). Concrete impli-
cations and research directions can be connected to various
phases of the BPM lifecycle model (Vidgof et al. 2023).
For example, in the context of process discovery, genera-
tive AI models  could  be  used  to  generate  process
descriptions, which can help  businesses identify and
understand the different stages of a process (Kecht et al.
2023). From the perspective of business process improve-
ment, generative process models could be used for idea
generation and to support innovative process (re-)design
initiatives (van Dun et al. 2023). In this regard, there is
great potential for generative AI to contribute to both
exploitative as well as explorative BPM design strategies
(Grisold et al. 2022). In addition, natural language pro-
cessing tasks related to BPM such as process extraction
from text could beneﬁt from generative AI without further
ﬁne-tuning using prompt engineering (Busch et al. 2023).
Likewise, other phases can beneﬁt owing to generative
AI’s ability to learn complex and non-linear relationships
in dynamic business processes  that can  be  used  for
implementation as well as in simulation and predictive
process monitoring among other things.
In the short term, robotic process automation (van der
Aalst et al. 2018; Herm et al. 2021) will beneﬁt as formerly
handcrafted processing rules can not only be replaced, but
entirely new types of automation can be enabled by ret-
roﬁtting and thus intelligentizing legacy software. In the
long run, we also see a large potential to support the phase
of business process execution in traditional BPM. Speciﬁ-
cally, we anticipate the development of a new generation of
process guidance systems. While traditional system designs
are based on static and manually-crafted knowledge bases
(Morana et al. 2019), more dynamic and adaptive systems
are feasible on the basis of large enterprise-wide trained
language models. Such systems could improve knowledge
retrieval tasks from a wide variety of heterogeneous sour-
ces, including manuals, handbooks, e-mails, wikis, job
descriptions, etc. This opens up new avenues of research
into how unstructured  and  distributed organizational

S. Feuerriegel et al.: Generative AI, Bus Inf Syst Eng
Table 2 Examples of research questions for future BISE research on generative AI
BISE department                          Research questions (examples)
Business process management              How can generative AI assist in automating routine tasks?
                                         How can generative AI reveal process innovation opportunities and support process (re-)design
                                         initiatives?
Decision analytics and data science      How can generative AI models be effectively ﬁne-tuned for domain-speciﬁc applications?
                                         How can the reliability of generative AI systems be improved?
Digital business management and digital  How can generative AI support managerial tasks such as resource allocation?
leadership                               How will the digital work of employees change with smart assistants powered by generative AI?
Economics of information systems         What are the welfare implications of generative AI?
                                         Which jobs and tasks are affected most by generative AI?
Enterprise modeling and enterprise       How can generative AI be used to support the construction and maintenance of enterprise models?
engineering                              How can generative AI support in enterprise applications (e.g., CRM, BI, etc.)?
Human computer interaction and social    How should generative AI systems be designed to foster trust?
computing                                What countermeasures are effective to prevent users from falling for AI-generated disinformation?
                                         To what extent can generative AI replace or augment crowdsourcing tasks?
                                         How can generative AI assist in education?
Information systems engineering and      What are effective design principles for developing generative AI systems?
technology                               How can generative AI support design science projects to foster creativity in the development of
                                         new IT artifacts?
knowledge can be incorporated into intelligent process          better ways how outputs can be veriﬁed (e.g., by offering
guidance systems.                                               additional explanations or references).
                                                                  Finally, questions arise about how generative AI can
4.2 Decision Analytics and Data Science                         natively support decision analytics and data science pro-
                                                                jects by closing the gap between modeling experts and
Despite the huge progress in recent years, several analytical   domain users (Zschech et al. 2020). For instance, it is
and technical questions around the development of gener-        commonly known that many AI models used in business
ative AI have yet to be solved. One open question relates to    analytics are difﬁcult to understand by non-experts (cf.
how   generative  AI  can  be  effectively customized   for     Senoner et al. 2022). As a remedy, generative AI could be
domain-speciﬁc   applications  and  thus  improve   perfor-     used to generate descriptions that explain the logic of
mance through higher degrees of contextualization. For          business analytics models and thus make the decision logic
example, novel and scalable techniques are needed to            more intelligible. One promising direction could be, for
customize conversational agents based on generative AI for      example, to use generative AI for translating post hoc
applications in medicine or ﬁnance. This will be crucial in     explanations derived from approaches like SHAP or LIME
practice to solve speciﬁc BISE-related tasks where cus-         into more intuitive textual descriptions or generate user-
tomization may bring additional performance gains. Novel        friendly descriptions of models that are intrinsically inter-
techniques for customization must be designed in a way          pretable (Slack et al. 2023; Zilker et al. 2023).
that ensures the safety of proprietary data and prevents the
data from being disclosed. Moreover, new frameworks are         4.3 Digital Business Management and Digital
needed for prompt engineering that are designed from a              Leadership
user-centered lens and thus promote interpretability and
usability.                                                      Generative AI has great potential to contribute to different
   Another important research direction is to improve the       types of value creation mechanisms, including knowledge
reliability of generative AI systems. For example, algo-        creation, task  augmentation,   and  autonomous    agency.
rithmic solutions are needed on how generative AI can           However, this also requires the necessary organizational
detect and mitigate hallucination. In addition to algorithmic   capabilities and conditions, where further research is nee-
solutions, more effort is also needed to develop user-cen-      ded to examine these ingredients more closely for the
tered solutions, that is, how users can reduce the risk of      context of generative AI to steer the technological possi-
falling for incorrect outcomes, for example, by developing      bilities in a successful direction (Shollo et al. 2022).
                                                                                                                 1 3

   That is, generative AI will lead to the development of
new business ideas, unseen product and service innova-
tions, and ultimately to the emergence of completely new
business models. At the same time, it will also have a
strong impact on intra-organizational aspects, such as work
patterns, organizational structures, leadership models, and
management practices. In this regard, we see that AI-based
assistant systems  previously  centered around   desktop
automation taking over more and more routine tasks such
as  event management,   resource  allocation, and  social
media account management to free up even more human
capacity (Maedche et al. 2019). Further, in algorithmic
management (Benlian et al. 2022; Cameron et al. 2023), it
should be examined how existing theories and frameworks
need to be contextualized or fundamentally extended in
light of the increasingly powerful capabilities of generative
AI.
   However, there are not only implications at the man-
agement level. The future of work is very likely to change
at all levels of an organization (Feuerriegel et al. 2022).
Due to the multi-modality of generative AI models, it is
conceivable that employees will work increasingly via
smart, speech-based interfaces, whereby the formulation of
prompts and the evaluation of their results could become a
key activity. Against this background, it is worth investi-
gating which new competencies are required to handle this
emerging technology (cf. Debortoli et al. 2014) and which
entirely new job proﬁles, such as prompt engineers, may
evolve in the near future (Strobelt et al. 2023).
   Generative AI is also expected to fundamentally reform
the  way   organizations manage,   maintain,  and  share
knowledge. Referring to the sketched vision of a new
process  guidance system  in  Sect. 4.1, we anticipate a
number of new opportunities for digital knowledge man-
agement, among others automated knowledge discovery
based on large amounts of unstructured distributed data
(e.g.,  identiﬁcation  of new   product   combinations),
improved knowledge sharing by automating the process of
creating, summarizing, and disseminating content (e.g.,
automated creation of wikis and FAQs in different lan-
guages), and personalized knowledge delivery to individual
employees based on their speciﬁc needs and preferences
(e.g., recommendations for speciﬁc training material).

4.4 Economics of Information Systems

Generative AI will have signiﬁcant economic implications
across various industries and markets. Generative AI can
increase efﬁciency and productivity by automating many
tasks that were previously performed by humans, such as
content creation, customer service, code generation, etc.
This can reduce costs and open up new opportunities for
growth   and  innovation  (Eloundou   et al. 2023).  For
1 3
         S. Feuerriegel et al.: Generative AI, Bus Inf Syst Eng
example, AI-based translation between different languages
is responsible for signiﬁcant economic gains (Brynjolfsson
et al. 2019). The BISE community can contribute by pro-
viding quantiﬁcation through rigorous causal evidence.
Given the velocity of AI research, it may be necessary to
take a more abstract problem view instead of a concrete
tool view. For example, BISE research could run ﬁeld
experiments to compare programmers with and without AI
support and thereby assess whether generative AI systems
for coding can improve the speed and quality of code
development. Similarly, researchers could test whether
generative AI will make artists more creative as they can
more easily create new content. A similar pattern was
previously observed for AlphaGo, which has led humans to
become better players in the board game Go (Shin et al.
2023).
Generative AI is likely to transform the industry as a
whole. This may hold true in the case of platforms that
make user-generated content available (e.g., shutterstock.-
com, pixabay.com, stackoverﬂow.com),  which  may   be
replaced by generative AI systems. Here, further research
questions arise as to whether the use of generative AI can
lead to a competitive advantage and how generative AI
changes competition. For example, what are the economic
implications if generative AI is developed as open-source
vs. closed-source systems? In this regard, a salient success
factor for the development of conversational agents based
on generative AI  (e.g., ChatGPT) are data from  user
interactions through dialogues and feedback on whether the
dialog was helpful. Hence, the value of such interaction
data is poorly understood and what it means if such data are
only available to a few Big Tech companies.
The digital transformation from generative AI also poses
challenges and opportunities for economic policy. It may
affect future work patterns and, indirectly, worker capa-
bility via restructured learning mechanisms. It may also
affect content sharing and distribution and, hence, have
non-trivial implications on the exploitation and protection
of intellectual properties. On top of that, a growing con-
centration of power over AI innovation in the hands of a
few companies may result in a monopoly of AI capabilities
and hamper future innovation, fair competition, scientiﬁc
progress, and thus welfare and human development at
large. All of these future impacts are important to under-
stand and provide meaningful directions for shaping eco-
nomic policy.

4.5 Enterprise Modeling and Enterprise Engineering

Enterprise models are important artifacts for capturing
insights into the core components and structures of an
organization, including business processes, resources,
information ﬂows, and IT systems (Vernadat 2020). A

S. Feuerriegel et al.: Generative AI, Bus Inf Syst Eng
major drawback of traditional enterprise models is that they
are static and may not provide the level of abstraction that
is required by the end user. Likewise, their construction
and maintenance are time-consuming and expensive and
require manual effort and human expertise (Silva et al.
2021). With generative AI, we see a large potential that
many of these limitations can be addressed by generative
AI  as assistive technology  (Sandkuhl  et al. 2018), for
example by automatically creating and updating enterprise
models at different levels of abstraction or generating
multi-modal representations.
   First empirical results suggest that generative AI is able
to generate useful conceptual models based on textual
problem   descriptions. Fill et al. (2023) show that ER,
BPMN, UML, and Heraklit models can not only be gen-
erated with very high to perfect accuracy from textual
descriptions, but they also explored the interpretation of
existing models and received good results. In the near
future, we  expect  more   research that  deals with  the
development,    evaluation,  and  application   of  more
advanced approaches. Speciﬁcally, we expect that learned
representations of enterprise models can be transformed
into more application-speciﬁc formats and can either be
enriched with further details or reduced to the essential
content.
   Against this background, the concept of ‘‘digital twins’’,
virtual representations of enterprise assets, may experience
new accentuation and extensions (Dietz and Pernul 2020).
Especially, in the public sector, where most organizational
assets are non-tangible in the form of deﬁned services,
speciﬁed procedures, legal texts, manuals, and organiza-
tional charts, generative AI can play a crucial role in dig-
itally mirroring and  managing   such  assets along their
lifecycles. Similar beneﬁts could be explored with physical
assets in Industry 4.0 environments (Lasi et al. 2014).
   In enterprise engineering, the role of generative AI
systems in existing as well as newly emerging IT land-
scapes to support the business goals and strategies of an
organization gives rise to numerous opportunities (e.g., in
ofﬁce solutions, customer relationship management and
business analytics applications, knowledge management
systems, etc.). Generative AI systems have the potential to
evolve into core enterprise applications that can either be
hosted on-premise or rented in the cloud. Unsanctioned use
bears the risk that third-party applications will be used for
job-related tasks without explicit approval or even knowl-
edge of the organization. This phenomenon is commonly
known as shadow IT and theories and frameworks have
been proposed to explain this phenomenon, as well as
recommending actions and policies to mitigate associated
risks (cf. Haag and Eckhardt 2017; Klotz et al. 2022). In
the light of generative AI, however, such approaches have
to be revisited for their applicability and effectiveness and,


if necessary, need to be extended. Nevertheless, this situ-
ation also offers the potential to explore and design new
approaches for more effective API  management    (e.g.,
including novel app store solutions, privacy and security
mechanisms, service level deﬁnitions, pricing, and licens-
ing models) so that generative  AI  solutions can  be
smoothly integrated into existing enterprise IT infrastruc-
tures without risking any unauthorized use and conﬁden-
tiality breaches.

4.6 Human Computer Interaction and Social
 Computing

Salient behavioral questions related to the interactions
between humans  and  generative AI  systems  are  still
unanswered. Examples  are related  to the  perception,
acceptance, adoption, and trust of systems using generative
AI. A study found that news was believed less if generated
by generative AI instead of humans (Longoni et al. 2022)
and another found that there is a replicant effect (Jakesch
et al. 2019). Such behavior is likely to be context-speciﬁc
and will vary by other antecedents highlighting the need for
a principled theoretical foundation to build successful
generative AI systems. The BISE   community   is well
positioned to develop rigorous design recommendations.
Further, generative AI is a key enabler for developing
high-quality interfaces for information systems based on
natural language that promote usability and accessibility.
For example, such interfaces will not only make interac-
tions more intuitive but will also facilitate people with
disabilities. Generative AI is likely to increase the ‘‘degree
of intelligence’’ of user assistance systems. However, the
design of effective interactions must also be considered
when increasing the degree of intelligence (Maedche et al.
2016). Similarly, generative AI will undoubtedly have an
impact on (computer-mediated) communication and col-
laboration, such as within companies. For example, gen-
erative AI can create optimized content for social media,
emails, and reports.  It can also help to improve  the
onboarding of new employees by creating personalized and
interactive training materials. It can also enhance collabo-
ration within teams by providing creative and intelligence
conservation agents that suggest, summarize, and synthe-
size information based on the context of the team (e.g.,
automated meeting notes).
Several applications  and  research opportunities  are
related to the use of generative AI in marketing and,
especially, e-commerce. It is expected that generative AI
can automate the creation of personalized marketing con-
tent, for instance, different sales slogans for introverts vs.
extroverts (Matz et al. 2017) or other personality traits as
personalized marketing content is more effective than a
one-content-ﬁts-all  approach  (Matz   et al.  2023).
                                            1 3

Generative AI may automate various tasks in marketing
and media where content generation is needed (e.g., writing
news stories, summarizing web pages for mobile devices,
creating thumbnail images for news stories, translating
written news to audio for blind people and Braille-sup-
ported formats for deaf people) that may be studied in
future research. Moreover, generative AI may be used in
recommender systems to boost the effectiveness of infor-
mation dissemination through personalization as content
can be tailored better to the abilities of the recipient.
   The education sector is another example that will need
to reinvent in some parts following the availability of
conversational agents (Kasneci et al. 2023; Gimpel et al.
2023). At ﬁrst glance, generative AI seems to constitute an
unauthorized aid that jeopardizes student grading so far
relying on written examinations and term papers. However,
over time, examinations will adapt, and generative AI will
enable the development of comprehensive digital teaching
assistants as well as the creation of supplemental teaching
material such as teaching cases and recap questions. Fur-
ther, the educator’s community will need to develop novel
guidelines  and  governance   frameworks    that educate
learners to rely appropriately on generative AI systems,
how to verify model outputs, and to engineer prompts
rather than the output itself.
   In addition, generative AI, speciﬁcally LLMs, can not
only be used to spot harmful content on social media (e.g.,
Maarouf et al. 2023), but it can also create realistic disin-
formation (e.g., fake news, propaganda) that is hard to
detect by humans (Kreps et al. 2022; Jakesch et al. 2023).
Notwithstanding, AI-generated disinformation has previ-
ously evolved as so-called deepfakes (Mirsky and Lee
2021), but recent advances in generative AI reduce the cost
of creating such disinformation and allow for unprece-
dented personalization. For example, generative AI can
automatically adapt the tone and narrative of misinforma-
tion to speciﬁc audiences that identify as extroverts or
introverts, left- or right-wing partisans, or people with
particular religious beliefs.
   Lastly, generative AI can facilitate—or even replace—
traditional crowdsourcing   where  annotations  or  other
knowledge tasks are handled by a larger pool of crowd
workers, for example in social media content annotation
(Gilardi et al. 2023) or market research on willingness-to-
pay for services and products (Brand et al. 2023). In gen-
eral, we expect that generative AI will automate many
other tasks being a zero-shot / few-shot learner. However,
this may also unfold negative implications: Users may
contribute less  to question-answering  forums   such  as
stackoverﬂow.com, which thus may reduce human-based
knowledge creation impairing the future performance of
AI-based   question-answering  systems   that  rely upon
human question-answering content for training. In a similar
1 3
          S. Feuerriegel et al.: Generative AI, Bus Inf Syst Eng
vein, the widespread availability of generative AI systems
may also propel research around virtual assistants. Previ-
ously, research made use of ‘‘Wizard-of-Oz’’ experiments
(Diederich et al. 2020), while future research may build
upon generative AI systems instead.
Crucially, automated content generation using genera-
tive AI is a new phenomenon, but automation in general
and how people are affected by automated systems has
been studied by scholars for decades. Thus, existing theo-
ries on the interplay of humans with automated systems
may be contextualized to generative AI systems.

4.7 Information Systems Engineering and Technology

Generative AI offers many engineering- and technology-
oriented research opportunities for the Information Systems
community as a design-oriented discipline. This includes
developing and evaluating design principles for generative
AI systems and  applications  to extend   the  limiting
boundaries of this technology (cf. Section 3). As such,
design principles can focus on how generative AI systems
can be made explainable to enable interpretability, under-
standing, and trust; how they can be designed reliable to
avoid discrimination effects or privacy issues; and how
they can be built more energy efﬁcient to promote envi-
ronmental sustainability (cf. Schoormann et al. 2023b).
While a lot of research is already being conducted in
technology-oriented disciplines such as computer science,
the BISE community can add its strength by looking at
design aspects through a socio-technical lens, involving
individuals, teams, organizations, and societal groups in
design activities, and thereby driving the ﬁeld forward with
new insights from a human–machine perspective (Maedche
et al. 2019).
Further, we see great potential that generative AI can be
leveraged to improve current practices in design science
research projects when constructing novel IT artifacts (see
Hevner et al. 2019). Here, one of the biggest potentials
could lie in the support of knowledge  retrieval tasks.
Currently, design knowledge  in  the  form  of  design
requirements, design principles, and design features is
often only available in encapsulated written papers or
implicitly embedded in instantiated artifacts. Generative AI
has the potential to extract such design knowledge that is
spread over a broad body of interdisciplinary research and
make it available in a collective form for scholars and
practitioners. This could also overcome the limitation that
design knowledge is currently rarely reused, which ham-
pers the fundamental idea of knowledge accumulation in
design science research (Schoormann et al. 2021).
Besides engineering actual systems and applications, the
BISE community should also investigate how generative
AI can be used to support creativity-based tasks when

S. Feuerriegel et al.: Generative AI, Bus Inf Syst Eng
initiating new design projects. In this regard, a promising       indicated otherwise in a credit line to the material. If material is not
direction could be to incorporate generative AI in design         included in the article’s Creative Commons licence and your intended
thinking and similar methodologies to combine human               use is not permitted by statutory regulation or exceeds the permitted
creativity  with   computational    creativity  (Hawlitschek      use, you will need to obtain permission directly from the copyright
2023). This may support different phases and steps of             holder. To view a copy of this licence, visit http://creativecommons.
                                                                  org/licenses/by/4.0/.
innovation projects, such as idea generation, user needs
elicitation, prototyping,  design   evaluation,  and   design
automation,   in which   different  types  of generative   AI     References
models and systems could be used and combined with each
other to form applications for creative industries (e.g.,         Agostinelli A, Denk TI, Borsos Z, Engel J, Verzetti M, Caillon A,
generated   user  stories with  textual  descriptions, visual            Huang Q, Jansen A, Roberts A, Tagliasacchi M, et al (2023)
mock-ups for user interfaces, and quick software proto-                  MusicLM: generating music from text. arXiv:2301.11325
                                                                  Asatiani A, Malo  P, Nagbøl  PR, Penttinen E, Rinta-Kahila T,
types for proofs-of-concept). If generative AI is used to co-            Salovaara A (2021) Sociotechnical envelopment of artiﬁcial
create  innovative  outcomes,   it may   also  enable  better            intelligence: an approach to organizational deployment of
reﬂection of the different design activities to ensure the               inscrutable artiﬁcial intelligence systems. J Assoc Inf Syst
necessary learning (Schoormann et al. 2023a).                            22(2):8
                                                                  Baird A, Maruping LM (2021) The next generation of research on IS
                                                                         use: a theoretical framework of delegation to and from agentic IS
                                                                         artifacts. MIS Q 45(1):315–341
5 Conclusion                                                      Baron-Cohen S (1997) Mindblindness: an essay on autism and theory
                                                                         of mind. MIT Press, Cambridge
                                                                  Benlian A,  Wiener  M,  Cram  WA,   Krasnova  H, Maedche   A,
Generative AI is a branch of AI that can create new content              ¨hlmann
                                                                         Mo     M, Recker J, Remus U (2022) Algorithmic manage-
such as texts, images, or audio that increasingly often                  ment. Bus Inf Syst Eng 64(6):825–839. https://doi.org/10.1007/
cannot be distinguished anymore from human craftsman-                    s12599-022-00764-w
ship. For this reason, generative AI has the potential to         Berente N, Gu B, Recker J, Santhanam R (2021) Special issue editor’s
                                                                         comments:  managing  artiﬁcial  intelligence.  MIS  Q
transform domains and industries that rely on creativity,                45(3):1433–1450
innovation, and knowledge processing. In particular, it           Beverungen D, Buijs JCAM, Becker J, Di Ciccio C, van der Aalst
enables new applications that were previously impossible                 WMP, Bartelheimer C, vom Brocke J, Comuzzi M, Kraume K,
or impractical for automation, such as realistic virtual                 Leopold H, Matzner M, Mendling J, Ogonek N, Post T, Resinas
                                                                         M, Revoredo K, del ´o-Ortega
                                                                                          Rı        A, La Rosa M, Santoro FM,
assistants, personalized education and service, and digital              Solti A, Song M, Stein A, Stierle M, Wolf V (2021) Seven
art. As such, generative AI has substantial implications for             paradoxes of business process management in a hyper-connected
BISE   practitioners  and  scholars as an   interdisciplinary            world. Bus Inf Syst Eng 63(2):145–156. https://doi.org/10.1007/
research community. In our Catchword article, we offered                 s12599-020-00646-z
                                                                  Birhane A, Prabhu VU, Kahembwe E (2021) Multimodal datasets:
a conceptualization of the principles of generative AI along             misogyny, pornography, and malignant stereotypes. arXiv:2110.
a model-, system-, and application-level view as well as a               01963
social-technical view and described limitations of current        Bishop C (2006) Pattern recognition and machine learning. Springer,
generative   AI.  Ultimately,  we   provided   an  impactful             New York
                                                                  Bommasani R, Hudson DA, Adeli E, Altman R, Arora S, von Arx S,
research agenda for the BISE community and thereby                       Bernstein MS, Bohg J, Bosselut A, Brunskill E, Brynjolfsson E,
highlight  the  manifold   affordances   that generative   AI            Buch S, Card D, Castellon R, Chatterji NS, Chen AS, Creel KA,
offers through the lens of the BISE discipline.                          Davis J, Demszky D, Donahue C, Doumbouya M, Durmus E,
                                                                         Ermon S, Etchemendy J, Ethayarajh K, Fei-Fei L, Finn C, Gale
Acknowledgements   During the preparation of this Catchword, we          T, Gillespie LE, Goel K, Goodman ND, Grossman S, Guha N,
contacted all current department editors at BISE to actively seek their  Hashimoto T, Henderson P, Hewitt J, Ho DE, Hong J, Hsu K,
feedback on our suggested directions. We gratefully acknowledge          Huang J, Icard TF, Jain S, Jurafsky D, Kalluri P, Karamcheti S,
their support.                                                           Keeling G, Khani F, Khattab O, Koh PW, Krass MS, Krishna R,
                                                                         Kuditipudi R, Kumar A, Ladhak F, Lee M, Lee T, Leskovec J,
                                                                         Levent  I, Li XL, Li X, Ma T,  Malik A,  Manning  CD,
Funding Open Access funding enabled and organized by Projekt             Mirchandani SP, Mitchell E, Munyikwa Z, Nair S, Narayan A,
DEAL.                                                                    Narayanan D, Newman B, Nie A, Niebles JC, Nilforoshan H,
                                                                         Nyarko JF, Ogut G, Orr L, Papadimitriou I, Park JS, Piech C,
Open Access   This article is licensed under a Creative Commons          Portelance E, Potts C, Raghunathan A, Reich R, Ren H, Rong F,
Attribution 4.0 International License, which permits use, sharing,       Roohani YH, Ruiz C, Ryan J, R’e C, Sadigh D, Sagawa S,
adaptation, distribution and reproduction in any medium or format, as    Santhanam K, Shih A, Srinivasan KP, Tamkin A, Taori R,
                                                                         Thomas AW,     `r
long as you give appropriate credit to the original author(s) and the              Trame  F, Wang RE, Wang W, Wu B, Wu J, Wu
source, provide a link to the Creative Commons licence, and indicate     Y, Xie SM, Yasunaga M, You J, Zaharia MA, Zhang M, Zhang
if changes were made. The images or other third party material in this   T, Zhang X, Zhang Y, Zheng L, Zhou K, Liang P (2021) On the
article are included in the article’s Creative Commons licence, unless
                                                                                                                     1 3

    opportunities and  risks of foundation  models.  arXiv:2108.
    07258https://doi.org/10.48550/arXiv.2108.07258
Brand J, Israeli A, Ngwe D (2023) Using GPT for market research.
    SSRN 4395751
Brown T, Mann B, Ryder N, Subbiah M, Kaplan JD, Dhariwal P,
    Neelakantan A, Shyam P, Sastry G, Askell A et al (2020)
    Language models are few-shot learners. Adv Neural Inf Process
    Syst 33:1877–1901
Brynjolfsson E, Hui X, Liu M (2019) Does machine translation affect
    international trade? Evidence from a large digital platform.
    Manag Sci 65(12):5449–5460
Burger B, Kanbach DK, Kraus S, Breier M, Corvello V (2023) On the
    use of AI-based tools like ChatGPT to support management
    research. Europ J Innov Manag 26(7):233–241. https://doi.org/
    10.1108/EJIM-02-2023-0156
Busch K, Rochlitzer1 A, Sola D, Leopold H (2023) Just tell me:
    Prompt engineering in business process management. arXiv:
    2304.07183
Caliskan A,  Bryson  JJ, Narayanan A   (2017) Semantics  derived
    automatically from language corpora contain human-like biases.

    Sci 356(6334):183–186
Cameron L, Lamers L, Leicht-Deobald U, Lutz C, Meijerink J,
       ¨hlmann
    Mo         M (2023) Algorithmic management: its implications

    for information systems research. Commun AIS 52(1):518–537.
    https://doi.org/10.17705/1CAIS.05221
Carlson SM, Koenig MA, Harms MB (2013) Theory of mind. WIREs
    Cogn Sci 4:391–402
C
¸ elikok MM, Peltola T, Daee P, Kaski S (2019) Interactive AI with a
    theory of mind. In: ACM CHI 2019 workshop: computational
    modeling in human-computer interaction, vol 80, pp 4215–4224
Chen L, Zaharia M, Zou J (2023) How is chatgpt’s behavior changing
    over time? arXiv:2307.09009
Chen M, Tworek J, Jun H, Yuan Q, Pinto HPdO, Kaplan J, Edwards

    H, Burda Y, Joseph N, Brockman G, et al (2021) Evaluating

    large language models trained on code. arXiv:2107.03374
Chiang T (2023) ChatGPT is a blurry JPEG of the web. https://www.
    newyorker.com/tech/annals-of-technology/chatgpt-is-a-blurry-
    jpeg-of-the-web, accessed 25 Aug 2023
Davison  RM,  Laumer   S, Tarafdar  M, Wong LHM      (2023)  ISJ
    editorial: pickled eggs: generative AI as research assistant or co-
    author? Inf Syst J Early View. https://doi.org/10.1111/isj.12455
De-Arteaga M, Feuerriegel S, Saar-Tsechansky M (2022) Algorith-
    mic fairness in business analytics: directions for research and
    practice. Prod Oper Manag 31(10):3749–3770
Debortoli S,   ¨ller
            Mu      O, vom Brocke J (2014) Comparing business
    intelligence and big data skills. Bus Inf Syst Eng 6(5):289–300.
    https://doi.org/10.1007/s12599-014-0344-2
                         ¨llner
Dellermann D, Ebel P, So       M, Leimeister JM (2019) Hybrid
    intelligence. Bus Inf Syst Eng 61(5):637–643. https://doi.org/10.
    1007/s12599-019-00595-2
Devlin J, Chang MW, Lee K, Toutanova K (2018) BERT: Pre-
    training of deep bidirectional transformers for language under-
    standing. arXiv:1810.04805
Diederich S, Brendel AB, Kolbe LM (2020) Designing anthropo-
    morphic enterprise conversational agents. Bus Inf Syst Eng
    62(3):193–209
Dietz M, Pernul G (2020) Digital twin: empowering enterprises
    towards  a  system-of-systems approach.  Bus  Inf  Syst Eng
    62(2):179–184. https://doi.org/10.1007/s12599-019-00624-0
Dolata M, Feuerriegel S, Schwabe G (2022) A sociotechnical view of
    algorithmic fairness. Inf Syst J 32(4):754–818
van Dun C, Moder L, Kratsch W,    ¨glinger
                                Ro        M (2023) ProcessGAN:
    supporting the creation of business process improvement ideas
    through  generative  machine  learning. Decis  Support  Syst
    165(113):880. https://doi.org/10.1016/j.dss.2022.113880

1 3
         S. Feuerriegel et al.: Generative AI, Bus Inf Syst Eng
Dwivedi YK, Kshetri N, Hughes L, Slade EL, Jeyaraj A, Kar AK,
 Baabdullah AM, Koohang A, Raghavan V, Ahuja M et al (2023)
 ‘‘So what if ChatGPT wrote it?’’ Multidisciplinary perspectives
 on opportunities, challenges and implications of generative
 conversational AI for research, practice and policy. Int J Inf
 Manag 71(102):642
Eloundou T, Manning S, Mishkin P, Rock D (2023) GPTs are GPTs:
 an early look at the labor market impact potential of large
 language models. arxiv:2303.10130, accessed 03 April 2023
Eskreis-Winkler L, Fishbach A (2022) Surprised elaboration: when
 white men  get  longer sentences. J  Personal Soc  Psychol
 123:941–956
Ferrara E (2023) Should ChatGPT be biased? Challenges and risks of
 bias in large language models. arXiv:2304.03738
Feuerriegel S, Dolata M, Schwabe G (2020) Fair AI: challenges and
 opportunities. Bus Inf Syst Eng 62:379–384
Feuerriegel S, Shrestha YR, von Krogh G, Zhang C (2022) Bringing
 artiﬁcial intelligence to business management. Nat Machine
 Intell 4(7):611–613
Fill HG, Fettke P, ¨pke
              Ko     J (2023) Conceptual modeling and large
 language models:  impressions from  ﬁrst  experiments with
 ChatGPT. EMISAJ 18(3):1–15. https://doi.org/10.18417/emisa.

 18.3
Ganguli D, Askell A, Schiefer N, Liao T,   ˇiu _
                                      Lukos ¯te K, Chen A,
 Goldie A, Mirhoseini A, Olsson C, Hernandez D, et al (2023)
 The capacity for moral self-correction in large language models.
 arXiv:2302.07459
Garcia T (2023) David Guetta replicated Eminem’s voice in a song

 using  artiﬁcial  intelligence.  https://variety.com/2023/music/
 news/david-guetta-eminem-artiﬁcial-intelligence-1235516924/,
 accessed 25 Aug 2023
Gilardi F, Alizadeh M, Kubli M (2023) ChatGPT outperforms crowd-
 workers for text-annotation tasks. arXiv:2303.15056
Gimpel H, Hall K, Decker S, Eymann T, ¨mmermann     ¨dche
                                 La            L, Ma     A,
  ¨glinger
 Ro       M, Ruiner C, Schoch M, Schoop M, et al (2023)
 Unlocking the power of generative ai models and systems such
 as GPT-4 and ChatGPT for higher education. https://digital.uni-
 hohenheim.de/ﬁleadmin/einrichtungen/digital/Generative_AI_
 and_ChatGPT_in_Higher_Education.pdf, accessed 25 Aug 2023
Goldman Sachs (2023) Generative AI could raise global GDP by 7%.
 https://www.goldmansachs.com/insights/pages/generative-ai-
 could-raise-global-gdp-by-7-percent.html
Goodfellow I, Pouget-Abadie J, Mirza M, Xu B, Warde-Farley D,
 Ozair S, Courville A, Bengio Y (2014) Generative adversarial
 nets. Adv Neural Inf Process Syst 27:2672–2680

Gray HM, Gray K, Wegner DM     (2007)  Dimensions  of  mind
 perception. Sci 315(5812):619–619
Grisold T, Groß S, Stelzl K, vom Brocke J, Mendling J, ¨glinger
                                               Ro        M,
 Rosemann M (2022) The ﬁve diamond method for explorative
 business process management. Bus Inf Syst Eng 64(2):149–166.
 https://doi.org/10.1007/s12599-021-00703-1
Haag S, Eckhardt A (2017)  Shadow   IT. Bus   Inf Syst  Eng
 59(6):469–473. https://doi.org/10.1007/s12599-017-0497-x
Haase J, Hanel PHP (2023) Artiﬁcial muses: generative artiﬁcial
 intelligence chatbots have risen to human-level creativity. arXiv:
 2303.12003
Hartmann J, Bergner A, Hildebrand C (2023a) MindMiner: uncov-
 ering linguistic markers of mind perception as a new lens to
 understand consumer-smart object relationships. J Consum Psy-
 chol. https://doi.org/10.1002/jcpy.1381
Hartmann J, Schwenzow J, Witte M (2023b) The political ideology of
 conversational AI: converging evidence on  ChatGPT’s   pro-

 environmental, left-libertarian orientation. arXiv:2301.01768
Hawlitschek F (2023) Interview with Samuel Tschepe on ‘‘Quo vadis
 design thinking?’’. Bus Inf Syst Eng 65(2):223–228. https://doi.
 org/10.1007/s12599-023-00792-0

S. Feuerriegel et al.: Generative AI, Bus Inf Syst Eng
Herm LV, Janiesch C, Reijers HA, Seubert F (2021) From symbolic
     RPA to intelligent RPA: challenges for developing and operating

     intelligent software robots.  In: International conference on
     business process management, pp 289–305
Hevner  A,  vom  Brocke  J, Maedche A    (2019)  Roles  of digital
     innovation in  design science  research. Bus  Inf Syst  Eng
     61(1):3–8. https://doi.org/10.1007/s12599-018-0571-z
        Ho J, Jain A, Abbeel P (2020) Denoising diffusion probabilistic
     models. Adv Neural Inf Process Syst 33:6840–6851
           Jakesch M, French M, Ma X, Hancock JT, Naaman M (2019) AI-
     mediated communication: how the perception that proﬁle text
     was written by AI affects trustworthiness. In: Conference on
     human factors in computing systems (CHI)
        Jakesch M, Hancock JT, Naaman M (2023) Human heuristics for AI-
     generated  language   are  ﬂawed.   Proc   Natl   Acad   Sci
     120(11):e2208839
       Janiesch C, Zschech P, Heinrich K (2021) Machine learning and deep
     learning. Electron Market  31(3):685–695.  https://doi.org/10.
     1007/s12525-021-00475-2
      Ji Z, Lee N, Frieske R, Yu T, Su D, Xu Y, Ishii E, Bang YJ, Madotto
     A, Fung P (2023) Survey of hallucination in natural language
     generation. ACM Comput Surv 55(12):1–38
Kasneci E, Seßler K,    ¨chemann
                      Ku           S, Bannert M, Dementieva D,
     Fischer F, Gasser U, Groh G, ¨nnemann       ¨llermeier
                                 Gu         S, Hu          E et al
     (2023) ChatGPT for good? On opportunities and challenges of
     large language  models  for education. Learn  Individ Differ
     103(102):274
Kecht C, Egger A, Kratsch W,      ¨glinger
                                 Ro        M (2023) Quantifying
     chatbots’  ability  to  learn  business  processes.  Inf  Syst
     113(102):176. https://doi.org/10.1016/j.is.2023.102176
     Khan J (2021) AI’s carbon footprint is big, but easy to reduce, Google
     researchers say. Fortune
          Kingma DP, Welling M (2013) Auto-encoding variational Bayes.
     https://doi.org/10.48550/arXiv.1312.6114
      Klotz S, Westner M, Strahringer S (2022) Critical success factors of
     business-managed IT: it takes two to tango. Inf Syst Manag
     39(3):220–240
       Kraus M, Feuerriegel S, Oztekin A (2020) Deep learning in business
     analytics and operations research: models,  applications and
     managerial implications. Europ J Oper Res 281(3):628–641.
     https://doi.org/10.1016/j.ejor.2019.09.018
Kreps S, McCain RM, Brundage M (2022) All the news that’s ﬁt to
     fabricate: AI-generated text as a tool of media misinformation.
     J Exp Polit Sci 9(1):104–117
   ¨gel
Kru    S, Ostermaier A, Uhl M (2023) ChatGPT’s inconsistent moral
     advice inﬂuences users’ judgment. Sci Report 13(1):4569
        Lasi H, Fettke P, Kemper HG, Feld T, Hoffmann M (2014) Industry
     4.0. Bus Inf Syst Eng 6(4):239–242. https://doi.org/10.1007/
     s12599-014-0334-4
         Li Y, Choi D, Chung J, Kushman N, Schrittwieser J, Leblond R,
     Eccles T, Keeling J, Gimeno F, Dal Lago A et al (2022)
     Competition-level code  generation with  alphacode.  Science
     378(6624):1092–1097
      Liu P, Yuan W, Fu J, Jiang Z, Hayashi H, Neubig G (2023) Pre-train,

     prompt, and predict: a systematic survey of prompting methods
     in natural language processing. ACM Comput Surv 55(9):1–35
           Longoni C, Fradkin A, Cian L, Pennycook G (2022) News from
     generative artiﬁcial intelligence is believed less. In: ACM
     conference  on  fairness,  accountability,  and  transparency

     (FAccT), pp 97–106
Maarouf A,   ¨r
            Ba  D, Geissler D, Feuerriegel S (2023) HQP: a human-
     annotated dataset for detecting online propaganda. arXiv:2304.
     14931
Maedche A, Morana S, Schacht S, Werth D, Krumeich J (2016)
     Advanced user assistance systems. Bus Inf Syst Eng 58:367–370


Maedche A, Legner C, Benlian A, Berger B, Gimpel H, Hess T, Hinz
 O, Morana S, ¨llner
            So      M (2019) AI-based digital assistants:
 opportunities, threats, and research perspectives. Bus Inf Syst
 Eng 61(4):535–544. https://doi.org/10.1007/s12599-019-00600-
 8
Matz S, Teeny J, Vaid SS, Harari GM, Cerf M (2023) The potential of
 generative AI for personalized persuasion at scale. PsyArXiv
Matz SC, Kosinski M, Nave G, Stillwell DJ (2017) Psychological
 targeting as an effective approach to digital mass persuasion.
 Proc Natl Acad Sci 114(48):12,714-12,719
Metz C (2023) Instant videos could represent the next leap in A.I.
 technology.  https://www.nytimes.com/2023/04/04/technology/
 runway-ai-videos.html, accessed 25 Aug 2023
Mirsky Y, Lee W (2021) The creation and detection of deepfakes: a
 survey. ACM Comput Survey 54(1):1–41
Morana S, Maedche A, Schacht S (2019) Designing process guidance
 systems. J Assoc Inf Syst pp 499–535, https://doi.org/10.17705/
 1jais.00542
Ng A, Jordan M (2001) On discriminative vs. generative classiﬁers: a
 comparison of logistic regression and naive Bayes. In: Advances
 in Neural Information Processing Systems, vol 14, pp 841–848,
 https://papers.nips.cc/paper_ﬁles/paper/2001/hash/

 7b7a53e239400a13bd6be6c91c4f6c4e-Abstract.html,  accessed

 25 Aug 2023
OpenAI  (2022)  Introducing ChatGPT.  https://openai.com/blog/
 chatgpt, accessed 25 Aug 2023
OpenAI (2023a) GPT-4 technical report. arXiv:2303.08774
OpenAI (2023b) How should AI systems behave, and who should

 decide? https://openai.com/blog/how-should-ai-systems-behave,
 accessed 25 Aug 2023
Park JS, O’Brien JC, Cai CJ, Morris MR, Liang P, Bernstein MS
 (2023) Generative agents: interactive simulacra of human
 behavior. arXiv:2304.03442
Peres R, Schreier M, Schweidel D, Sorescu A (2023) On ChatGPT
 and beyond: how generative artiﬁcial intelligence may affect
 research, teaching, and practice. Int J Res Market 40:269–275
Rabinowitz NC, Perbet F, Song HF, Zhang C,  Eslami  SMA,
 Botvinick MM (2018) Machine theory of mind. In: International
 conference on machine learning, PMLR, vol 80, pp 4215–4224,
 http://proceedings.mlr.press/v80/rabinowitz18a.html,  accessed
 25 Aug 2023
Rai A (2020) Explainable AI: from black box to glass box. J Acad
 Market Sci 48:137–141
Ramaswamy V, Ozcan K (2018) What is co-creation? An interac-
 tional creation framework and its implications for value creation.

 J Bus Res 84:196–205
Reisenbichler M, Reutterer T, Schweidel DA, Dan D  (2022)
 Frontiers: supporting content marketing with natural language
 generation. Market Sci 41(3):441–452
Rombach R, Blattmann A, Lorenz D, Esser P, Ommer B (2022) High-
 resolution image synthesis with latent diffusion models. In:
 IEEE/CVF conference on computer vision and pattern recogni-
 tion, pp 10684–10695
Sandkuhl K, Fill H, Hoppenbrouwers S, Krogstie J, Matthes F,
 Opdahl AL, Schwabe G, Uludag   ¨
                               O, Winter R (2018) From
 expert discipline to common practice: a vision and research
 agenda for extending the reach of enterprise modeling. Bus Inf
 Syst Eng 60(1):69–80. https://doi.org/10.1007/s12599-017-0516-
 y
Schoormann T,  ¨ller
        Mo     F, Hansen MRP (2021) How do researchers
 (re-)use design principles: An inductive analysis of cumulative

 research. In: The Next Wave of Sociotechnical Design, Springer,
 Cham, Lecture Notes in Computer Science, pp 188–194, https://
 doi.org/10.1007/978-3-030-82405-1_20

                                               1 3

                                                                                   S. Feuerriegel et al.: Generative AI, Bus Inf Syst Eng
Schoormann           ¨nder
            T, Stadtla    M, Knackstedt R (2023) Act and reﬂect:        Susarla A, Thatcher RGJB, Sarker S (2023) Editorial: the janus effect
     integrating reﬂection into design thinking. J Manag Inf Syst         of generative AI: charting the path for responsible conduct of
     40(1):7–37. https://doi.org/10.1080/07421222.2023.2172773            scholarly  activities in information systems.  Inf Syst  Res
Schoormann T, Strobel G,     ¨ller
                           Mo     F, Petrik D, Zschech P (2023)           34(2):399–408. https://doi.org/10.1287/isre.2023.ed.v34.n2
     Artiﬁcial intelligence for sustainability: a systematic review of  Sutskever I, Vinyals O, Le QV (2014) Sequence to sequence learning
     information systems literature. Commun AIS 52(1):8                   with   neural  networks.   Adv   Neural   Inf  Process  Syst
Schramowski P, Turan C, Andersen N, Rothkopf CA, Kersting K               27:3104–3112
     (2022) Large pre-trained language models contain human-like        Teubner T, Flath CM, Weinhardt C, van der Aalst W, Hinz O (2023)
     biases of what is right and wrong to do. Nat Machine Intell          Welcome to the era of ChatGPT. Bus Inf Syst Eng 65(2):95–101.
     4(3):258–268                                                         https://doi.org/10.1007/s12599-023-00795-x
Schwartz  R, Dodge   J, Smith  NA,  Etzioni O  (2020)  Green AI.        Unsal S, Atas H, Albayrak M, Turhan K, Acar AC,   ˘an
                                                                                                                       Dog    T (2022)
     Commun ACM 63(12):54–63                                              Learning functional properties of proteins with language models.
    ¨bel
Scho    S, Schmitt A, Benner D, Saqr M, Janson A, Leimeister JM           Nat Machine Intell 4(3):227–245
     (2023) Charting  the evolution and  future of conversational       van der Aalst WMP, Bichler M, Heinzl A (2018) Robotic process
     agents: a research agenda along ﬁve waves and new frontiers. Inf     automation. Bus Inf Syst Eng 60(4):269–272. https://doi.org/10.
     Syst Front. https://doi.org/10.1007/s10796-023-10375-9               1007/s12599-018-0542-4
Senoner  J, Netland  T, Feuerriegel  S (2022)  Using  explainable       Vaswani A, Shazeer N, Parmar N, Uszkoreit J, Jones L, Gomez AN,
     artiﬁcial intelligence to improve process quality: evidence from     Kaiser Ł, Polosukhin I (2017) Attention is all you need. Adv
     semiconductor manufacturing. Manag Sci 68(8):5704–5723               Neural Inf Process Syst 30:6000–6010
Shin M, Kim J, van Opheusden B, Grifﬁths TL (2023) Superhuman           Vernadat F (2020) Enterprise modelling: research review and outlook.
     artiﬁcial intelligence can improve human decision-making by          Comput Indust 122(103):265. https://doi.org/10.1016/j.compind.
     increasing novelty. Proc Natl Acad Sci 120(12):e2214840,120          2020.103265
Shollo A, Hopf K, Thiess T,     ¨ller
                              Mu     O (2022) Shifting ML value         Vidgof M, Bachhofner S, Mendling J (2023) Large language models
     creation mechanisms: a process model of ML value creation.           for business process management: opportunities and challenges.
     J Strateg Inf Syst 31(3):101,734. https://doi.org/10.1016/j.jsis.    In: Business  process management    forum. Lecture  Notes  in
     2022.101734                                                          Computer Science, Springer, Cham, pp 107-123
Siebers P, Janiesch C, Zschech P (2022) A survey of text represen-      von Zahn M, Feuerriegel S, Kuehl N (2022) The cost of fairness in
     tation methods and their genealogy. IEEE Access 10:96,492-           AI: evidence from e-commerce. Bus Inf Syst Eng 64:335–348
     96,513. https://doi.org/10.1109/ACCESS.2022.3205719                Wolfe R, Banaji MR, Caliskan A (2022) Evidence for hypodescent in
Silva N, Sousa P, Mira da Silva M (2021) Maintenance of enterprise        visual semantic AI. In: ACM conference on fairness, account-
     architecture models. Bus Inf Syst Eng 63(2):157–180. https://doi.    ability, and transparency, pp 1293–1304
     org/10.1007/s12599-020-00636-1                                     Ziegler DM, Stiennon N, Wu J, Brown TB, Radford A, Amodei D,
Slack D, Krishna S, Lakkaraju H, Singh S (2023) Explaining machine        Christiano P, Irving G (2019) Fine-tuning language models from
     learning models with interactive natural language conversations      human preferences. arXiv:1909.08593
     using TalkToModel. Nat Machine Intell 5:873–883                    Zilker S, Weinzierl S, Zschech P, Kraus M, Matzner M (2023) Best of
Smits J, Borghuis T (2022) Generative AI and intellectual property        both worlds: combining predictive power with interpretable and
     rights. Law and artiﬁcial intelligence: regulating AI and applying   explainable results for patient pathway prediction. In: Proceed-
     ai in legal practice. Springer, Heidelberg, pp 323–344               ings of the 31st European Conference on Information Systems
Spitale G, Biller-Andorno N, Germani F (2023) AI model GPT-3 (dis)        (ECIS), Kristiansand, Norway
     informs us better than humans. Sci Adv 9:eadh1850                  Zschech P, Horn     ¨schele
                                                                                       R, Ho        D, Janiesch C, Heinrich K (2020)
Strobelt H, Webson A, Sanh V, Hoover B, Beyer J, Pﬁster H, Rush           Intelligent user assistance for automated data mining method
     AM (2023) Interactive and visual prompt engineering for ad-hoc       selection. Bus Inf Syst Eng 62(3):227–247. https://doi.org/10.
     task adaptation with large language models. IEEE Transact            1007/s12599-020-00642-3
     Visual Comput Graphics 29(1):1146–1156. https://doi.org/10.
     1109/TVCG.2022.3209479
1 3
View publication stats