

{0}------------------------------------------------

# Make Identity Unextractable yet Perceptible: Synthesis-Based Privacy Protection for Subject Faces in Photos

Tao Wang, Yushu Zhang, Xiangli Xiao, Kun Xu, Lin Yuan, Wenying Wen, and Yuming Fang

**Abstract**—Deep learning-based face recognition (FR) technology exacerbates privacy concerns in photo sharing. In response, the research community developed a suite of anti-FR methods to block identity extraction by unauthorized FR systems. Benefiting from quasi-imperceptible alteration, perturbation-based methods are well-suited for privacy protection of *subject faces* in photos, as they allow familiar persons to recognize subjects via naked eyes. However, we reveal that perturbation-based methods provide a *false sense of privacy* through theoretical analysis and experimental validation. Therefore, new alternative solutions should be found to protect subject faces.

In this paper, we explore synthesis-based methods as a promising solution, whose challenge is to enable familiar persons to recognize subjects. To solve the challenge, we present a key insight—in most photo sharing scenarios, familiar persons recognize subjects through *identity perception* rather than *meticulous face analysis*. Based on the insight, we propose the first synthesis-based method dedicated to subject faces, i.e., PerceptFace, which can make identity unextractable yet perceptible. To enhance identity perception, a new perceptual similarity loss is designed for faces, reducing the alteration in regions of high sensitivity to human vision.

As a synthesis-based method, PerceptFace can inherently provide reliable *identity protection*. Meanwhile, out of the confine of *meticulous face analysis*, PerceptFace focuses on *identity perception* from a more practical scenario, which is also enhanced by the designed perceptual similarity loss. Sufficient experiments show that PerceptFace achieves a superior trade-off between identity protection and identity perception compared to existing methods. We provide a public API of PerceptFace and believe that it has great potential to become a practical anti-FR tool. Our code is available at <https://github.com/daizige/PerceptFace>.

**Index Terms**—Visual privacy, biometric protection, human perception, face recognition, adversarial perturbation.

## I. INTRODUCTION

PHOTO sharing on online social networks (OSNs) [1] is a crucial service, that has revolutionized the way people record lives and express emotions. Nevertheless, with the development of deep learning-based face recognition (FR) technology, such a service also raises privacy concerns, which

draws major attention from the research community and data protection agencies [2]. The faces in the shared photos can be captured by individuals or organizations to train FR models, automatically identified by FR systems for malicious purposes, or misused by DeepFake technologies [3] to enable face forgery. In addition, sensitive attributes exposed in identity can cause additional potential harm to individuals [4], [5], e.g., sex discrimination. Notoriously, Clearview crawled more than 3 billion images from OSNs without the consent or knowledge of individuals to build its large-scale FR system. For payment, anyone is permitted to monitor people by its FR system.

In response, the research community has developed a number of anti-FR methods [6], [7] for face privacy. Before uploading a photo to OSNs, users can process the faces in the photo with anti-FR methods, preventing the correct recognition by *unauthorized FR systems*. Leaving aside methods (e.g., encryption [8], [9] or transformation [10]–[12]) that compromise visual quality, existing anti-FR methods can be mainly categorized into two groups: 1) Synthesis-based methods generate a face with a new identity to replace the original face, thus *removing* the original identity; 2) Perturbation-based methods add quasi-imperceptible noise to disturb the judgment of FR systems, thus *concealing* the original identity.

Perturbation-based methods differ from synthesis-based methods in one major characteristic—they *allow human observers to see the original identity, while synthesis-based methods block it*. This characteristic helps perturbation-based methods particularly well-suited for protecting face privacy of *subjects* (distinct from bystanders [13]) in photos, which still enables recognition by familiar friends with naked eyes, thereby preserving the photo utility without compromising its social function. A variety of perturbation-based methods [14]–[16] were developed to assist individuals in protecting subject faces, e.g., Fawkes [17] (pixel level) and ATM-GAN [18] (semantic level). Fawkes has released public software versions for practical applications, while most methods are currently limited to the research prototype stage.

**Motivation.** It is imperative to sound the alarm for the research community that—*perturbation-based methods provide just a false sense of privacy*. One compelling reason is the inherent unsustainability: they defend against privacy breaches by attacking FR systems to discover vulnerabilities [19], but these vulnerabilities would be fixed by ever-evolving FR systems, which makes the protection just temporary. It indicates that *an attack is just an attack and not the best defense*. Thus, the research community should find new alternative solutions

Tao Wang and Kun Xu are with the College of Computer Science and Technology, Nanjing University of Aeronautics and Astronautics, Nanjing 211106, China (e-mail: {wangtao21, xukun930}@nuaa.edu.cn).

Yushu Zhang, Wenying Wen, Xiangli Xiao, and Yuming Fang are with the School of Computing and Artificial Intelligence, Jiangxi University of Finance and Economics, Nanchang 330032, China (e-mail: zhangyushu@jxufe.edu.cn, xiaoxiangli@jxufe.edu.cn; wenyingwen@sina.cn; fa0001ng@e.ntu.edu.sg).

Lin Yuan is with Chongqing Key Laboratory of Image Cognition, Chongqing University of Posts and Telecommunications, Chongqing 400065, China (e-mail: yuanlin@cqupt.edu.cn).

{1}------------------------------------------------

![Figure 1: Illustration of the proposed PerceptFace. The diagram shows a workflow starting from an 'Original Photo' (AI-enlarged by Picsman). It branches into two paths: 'Unreliable protection' (perturbation-based methods like Fawkes and AMT-GAN) and 'Low perceptual similarity' (synthesis-based methods like RiDDLE and Disguise). Both paths lead to a 'Protected Photo'. A 'PerceptFace' block is shown between the 'Cropped Face' and 'Protected Face'. A graph on the right plots 'Identity Perception (Utility)' against 'Identity Protection (Privacy)'. It shows 'Existing perturbation-based methods' (green dots) and 'Existing synthesis-based methods' (yellow dots) clustered at low utility and low privacy. 'PerceptFace' (blue triangle) is positioned at high utility and high privacy. A red 'X' marks 'Extract identity by FR systems' (failing) and a blue checkmark marks 'Perceive identity by familiar persons' (succeeding).](aa9e46d6f962be5cebcbb5c654c9b13e_img.jpg)

Figure 1: Illustration of the proposed PerceptFace. The diagram shows a workflow starting from an 'Original Photo' (AI-enlarged by Picsman). It branches into two paths: 'Unreliable protection' (perturbation-based methods like Fawkes and AMT-GAN) and 'Low perceptual similarity' (synthesis-based methods like RiDDLE and Disguise). Both paths lead to a 'Protected Photo'. A 'PerceptFace' block is shown between the 'Cropped Face' and 'Protected Face'. A graph on the right plots 'Identity Perception (Utility)' against 'Identity Protection (Privacy)'. It shows 'Existing perturbation-based methods' (green dots) and 'Existing synthesis-based methods' (yellow dots) clustered at low utility and low privacy. 'PerceptFace' (blue triangle) is positioned at high utility and high privacy. A red 'X' marks 'Extract identity by FR systems' (failing) and a blue checkmark marks 'Perceive identity by familiar persons' (succeeding).

Fig. 1. Illustration of the proposed PerceptFace. PerceptFace protects the subject face in the photo, making the identity unextractable yet perceptible, where the original photo is AI-enlarged by Picsman, avoiding potential ethical and copyright issue. PerceptFace achieves a superior trade-off between privacy and utility compared to existing perturbation-based (unreliable protection) and synthesis-based methods (low perceptual similarity).

to protect subject face, which was also highlighted in the recent systematization of knowledge (SoK) work [6].

**Our Work.** In this paper, we explore a promising solution (synthesis-based methods) toward the development of more practical anti-FR tools for subject faces. Due to the change to a new identity, synthesis-based methods cannot be obstructed by the ever-evolution of FR systems. However, since facial identity and appearance are strongly correlated, replacing a new identity would inherently result in a new facial appearance, which blocks human vision from recognizing subjects. For this, we attempt to solve the challenge of synthesis-based methods in protecting subject face privacy, i.e., enabling familiar persons to recognize subjects.

Concretely, the major contributions of our work are summarized as follows:

- **Important Alert.** We reveal that perturbation-based methods provide just a *false sense of privacy*. In theoretical analysis, we point out four aspects, including inherent unsustainability, unattainable transferability, weak robustness, and wrong priority. In experimental validation, we use common noise and commercial APIs to verify that the protection of mainstream methods is unreliable. Meanwhile, we alert the research community about the cautious utilization of adversarial perturbation for protecting privacy.
- **Key Insight.** We present a key insight based on existing research [20], [21] and life experience, which is more aligned with the real-world scenario—*In most photo sharing scenarios, the recognition of subjects relies on identity perception rather than meticulous face analysis by familiar persons*. This insight emphasizes “the photo containing faces” from a global perspective, breaking the limitation of existing methods to consider “faces in the photo” from a local perspective. We further divide the process of identity perception into *contextual perception of non-facial regions* and *coarse-grained perception of facial regions*.
- **Promising Method.** Based on the above insight, we propose **the first synthesis-based method dedicated to subject faces**, i.e., PerceptFace, which can allow familiar persons to perceive the subject identity while preventing FR systems from extracting it. As shown in Fig. 1, PerceptFace achieves a superior trade-off between

privacy and utility compared to existing perturbation-based (Fawkes [17] and AMT-GAN [18]) subject to the *unreliable protection*, and synthesis-based methods (RiDDLE [22] and Disguise [23]) subject to the *low perceptual similarity*. According to the process of identity perception, PerceptFace benefits from *attribute-preserved identity manipulation* and *perception-enhanced identity transformation*.

- **Technical Innovation.** We design an innovative perceptual similarity loss for faces, which promotes PerceptFace to achieve better identity perception than existing synthesis-based methods [22], [23]. Existing methods apply the perceptual similarity loss (mainly LPIPS) just targeted at *generic categories* of images, which limits their ability to preserve identity perception. This is because human vision systems appear to allocate specialized neural resources to face perception [24]. More appropriately, our perceptual similarity loss is targeted at *face categories* of images, which can mimic the face perception of human vision by introducing perceptual sensitivity.

The remainder of the paper is organized as follows. Section II briefly introduces the related work. Section III presents the threat model and design goals. Section IV reveals the limitations of perturbation-based methods, and Section V shows our thoughts and insights. Section VI details the proposed method. Experimental results are presented in Section VII and followed by a conclusion in Section VIII.

## II. RELATED WORK

### 1) Encryption or Transformation-based Methods:

Encryption-based methods utilize authorized keys to encrypt and later restore images. Tajik *et al.* [8] developed a thumbnail-preserving image encryption scheme for cloud storage, allowing users to preview photos. ProShare [9] enables different regions of a photo to be encrypted and accessed based on user permissions. It securely encrypts images and embeds them directly within the JPEG file itself, providing users with more granular control over their online photo privacy.

In contrast, transformation-based methods do not support reversibility but offer higher efficiency. FaceObfuscator [10] removes key identity-related signals in the frequency domain

{2}------------------------------------------------

and adopts a carefully designed obfuscation mechanism to resist reconstruction attacks. It can quickly process facial images and remove any visual information except for identity information. CamPro [11] anonymizes facial regions at the hardware level by adjusting camera imaging parameters, providing protection against hijacking attacks. As the face has been erased during the imaging process, it provides strong privacy guarantees.

However, the protected results produced by these methods appear visually unnatural. This distinct visual discrepancy makes it easy for attackers to notice and distinguish between protected and unprotected faces. Consequently, the research community tends to favor the following two main categories of methods.

2) *Perturbation-based Methods*: Early perturbation-based methods [25]–[27] primarily relied on pixel-level noise to concealing identity. Fawkes [17] is a representative work in this category, offering multiple levels of noise intensity to meet diverse user privacy requirements. TIP-IM [15] introduces a maximum mean square error criterion to suppress unnecessary noise, improving visual quality. OPOM [28] generates a personalized face mask for each user, achieving a practical trade-off between protection effectiveness and computational efficiency. P3-Mask [29] builds on OPOM by integrating a focal diversity-optimized ensemble learning approach into the mask generation process, which significantly strengthens its robustness against unknown FR models.

Recent works have shifted towards semantic-level perturbations [30], [31], which provide better transferability and robustness. AMT-GAN [18] perturbs facial appearance via makeup synthesis to evade recognition. 3DAM-GAN [32] leverages facial symmetry to render realistic and robust makeup, thereby enhancing the transferability to black-box models. However, despite offering improved protection performance, the method introduces noticeable visual distortions in the nasal region. ImU [33] constructs style perturbations in the latent space of StyleGAN, while GIFT [34] operates in a better latent space of StyleGAN to generate global feature perturbations. DiffPrivate [35] further advances this direction by constructing global feature perturbations in the latent space of diffusion models (DMs), improving generalization under varied recognition settings. To enhance security, DivTrackee [36] advocates for explicitly promoting the diversity of protection results. This is achieved by building upon a text-guided image generation framework and a diversity loss based on the first-in, first-out (FIFO) principle.

3) *Synthesis-based Methods*: Early synthesis-based methods [37]–[39] typically replace the entire facial region to achieve anonymization. DeepPrivacy [40] removes the original face and employs the generative adversarial network to synthesize a new face. RiDDLE [22] introduces a key-controlled replacement mechanism, enabling both reversibility and diversity in face substitution. Lopez *et al.* [41] introduced a hardware-level method for face de-identification. Their approach first involves training an optical encoder to produce a privacy-preserving face heatmap, which is then used to generate a novel face based on a reference image.

To better preserve facial attributes, recent methods [42],

[43] focus on semantic-level manipulation of identity-specific features. CIAGAN [44] utilizes a conditional GAN to take identity labels as a conditional input, which allows it to generate controllable and diverse protection results. Additionally, it preserves the facial structure to ensure the protected faces remain detectable. FaceRSA [45] is a facial identity encryption framework that possesses all the properties of RSA. It achieves facial identity anonymization and secure de-anonymization within the decoupled StyleGAN latent space using a carefully designed cryptographic mapper. Disguise [23] transforms identity via differential privacy in a disentanglement-based framework, preventing re-identification attacks. AIDPro [46] uses authentication information as a condition to guide identity transformation, enabling robust image authentication. It achieves robustness to a wide range of image distortions, such as JPEG compression and screen shooting, without needing a noise layer.

## III. PRELIMINARY

### A. System Model and Threat Model

![Diagram illustrating the general photo sharing scenario and threat model. A User shares a Photo on OSNs (Online Social Networks). Friends can View the photo. An Adversary can Extract identity from the photo, leading to Unauthorized collection, Individual tracking, and Facial forgery. Bystanders and Subjects are also shown.](062ad684575a714449a7e040c0e1ec00_img.jpg)

The diagram illustrates a photo sharing scenario and a threat model. On the left, a 'User' icon is shown next to a 'Photo' icon. An arrow points from the 'Photo' to a cloud labeled 'OSNs' (Online Social Networks). From the 'OSNs' cloud, an arrow points to a 'Friends' icon, which is next to a 'View photo' icon. Another arrow points from the 'OSNs' cloud to an 'Adversary' icon. The 'Adversary' icon is next to an 'Extract identity' icon. A red box highlights the 'Extract identity' icon and the text 'Unauthorized collection', 'Individual tracking', and 'Facial forgery'. Below the 'Adversary' icon, there are icons for 'Bystanders' and 'Subjects'.

Diagram illustrating the general photo sharing scenario and threat model. A User shares a Photo on OSNs (Online Social Networks). Friends can View the photo. An Adversary can Extract identity from the photo, leading to Unauthorized collection, Individual tracking, and Facial forgery. Bystanders and Subjects are also shown.

Fig. 2. General photo sharing scenario and threat model.

We consider a general photo sharing scenario, in which a user shares a photo containing faces on online social networks (OSNs) and then friends can view the photo to feel the conveyed message. During the sharing process, photos may undergo compression, re-encoding, or other image processing operations by the social platform.

Our threat model focuses on **identity extraction by FR tools**, which can be used for malicious purposes, e.g., unauthorized collection, individual tracking, or facial forgery, as shown in Fig. 2. Given the widespread availability of high-performance pre-trained FR tools (advanced models or commercial APIs), adversaries can readily access them to facilitate such a malicious goal.

In our work, identity inference by professional attackers is not considered. Although some non-FR tools can infer the subject's identity based on contextual cues such as location or clothing, they are incapable of obtaining facial information. Moreover, lacking direct visual evidence, such identity inference has low accuracy and credibility.

### B. Face Category and Anti-FR Methods

We classify the faces to be protected in photos, as they have different protection requirements.

- *Subject Face*: Subjects are the main persons in a photo. The subjects often occupy a prominent position in the photo and maybe the user themselves, a close friend, or a family member. Subject faces are not allowed to be significantly modified, because friends care who the subject is.

{3}------------------------------------------------

- *Bystander Face*: Bystanders [13] are other people who appear unintentionally in a photo. They are not the subject of the photo and maybe casual passers-by, other participants in an event, or strangers in a public place. Bystander faces are allowed to be significantly modified because friends do not care who the bystander is.

Setting aside methods that degrade visual quality [47]–[49], existing anti-FR methods can be mainly categorized:

- *Perturbation-based Method*: Such methods add quasi-imperceptible noise to disturb the judgment of FR models, thus concealing the original identity. The noise can be pixel level (e.g., Fawkes [17]) or semantic level (e.g., ATM-GAN [18]). Since this noise has a weak effect on the facial appearance, human vision can observe the original identity.
- *Synthesis-based Method*: Such methods generate a face with a new identity to replace the original face, thus removing the original identity. The generated face can be entirely new (e.g., RiDDLE [22]) or only altered in identity (e.g., Disguise [23]). Since a new identity brings a new facial appearance, human vision cannot observe the original identity.

Based on the above, perturbation-based methods are applicable to protect subject face privacy, and synthesis-based methods are applicable to protect bystander face privacy, which are visually displayed in Fig. 3. Distinctively, our method is the *first synthesis-based method*, which is designed for subject face privacy rather than bystander face.

![Figure 3: Diagram illustrating anti-FR methods for subject and bystander faces. For subject faces, perturbation-based methods are applied, resulting in a 'Protected face' that human vision can still identify as the original person, while machine vision identifies it as 'other'. For bystander faces, synthesis-based methods are applied, resulting in a 'Protected face' that both human and machine vision identify as 'other'.](12a6537c92844d5b393104c02e8dfc2f_img.jpg)

Figure 3: Diagram illustrating anti-FR methods for subject and bystander faces. For subject faces, perturbation-based methods are applied, resulting in a 'Protected face' that human vision can still identify as the original person, while machine vision identifies it as 'other'. For bystander faces, synthesis-based methods are applied, resulting in a 'Protected face' that both human and machine vision identify as 'other'.

Fig. 3. Corresponding anti-FR Methods applicable to different categories of faces. Perturbation-based methods are applied to subject faces, and synthesis-based methods are applied to bystander faces. Distinctly, our method is the *first synthesis-based method* dedicated to subject faces.

### C. Design Goals

Our work is to design an anti-FR tool to protect *subject faces* in photos, as we reveal that perturbation-based methods cannot reliably protect them in Section IV. Without destroying non-cropped areas in the photo, the design goals are as follows:

**Goal ①: Preserving the utility that familiar persons can recognize the subject via human vision.** Anti-FR tools should not compromise the social utility of photos. Although these tools modify only the *cropped facial region*, such alterations may hinder friends from visually recognizing the subject, thereby impairing the intended usage of the photo in social contexts. Thus, it is essential to allow recognition by familiar persons.

**Goal ②: Protecting the privacy that FR systems cannot extract identity of the subject via machine vision.** Facial identity is highly sensitive, and its leakage can pose severe privacy risks. With the high accuracy and wide accessibility of deep learning-based FR tools, facial identity is increasingly vulnerable to unauthorized extraction. Thus, it is essential to prevent identity extraction by FR systems.

### D. Dodging or Impersonation Protection

According to whether the protected identity is specific, existing protection methods can be categorized into *dodging protection* and *impersonation protection*. The former enables the protected identity to be a *non-targeted identity*, while the latter enables it to be a *targeted identity*.

Our work focuses on *dodging protection* and we argue that impersonation protection is less suitable in practice:

1) *Risk of Re-Identification*: Prior research [50] has shown that high impersonation success may coincide with low dodging success. This implies that even if the protected identity is close to the target identity, it may also remain highly similar to the original identity. As a result, adversaries may still re-identify the original individual by selecting the top-2 most similar identities. Therefore, impersonation protection may pose a significant privacy risk of re-identification.

2) *Misuse of Target Identity*: Impersonation protection requires specifying a target identity, which may correspond to a real individual for malicious purposes, raising concerns over portrait rights and ethical implications. For instance, a user or an insider might deliberately set a celebrity as the target identity and fabricate evidence implicating them in illegal activities. Even if such allegations are later refuted, the incident may cause reputational harm and public distress to the celebrity. Therefore, impersonation protection may raise ethical concerns regarding identity misuse.

## IV. PERTURBATION-BASED METHODS CANNOT RELIABLY PROTECT SUBJECT FACE PRIVACY

### A. Intuitive Effectiveness

![Figure 4: Diagram illustrating the intuitive effect of perturbation-based methods. A subject face is combined with a colorful adversarial perturbation (indicated by a plus sign). Human vision perceives the result as 'Look like no change' (with a blue checkmark), while machine vision perceives it as 'Different identity' (with a red checkmark).](320cb123e33258367abe408f763f5e83_img.jpg)

Figure 4: Diagram illustrating the intuitive effect of perturbation-based methods. A subject face is combined with a colorful adversarial perturbation (indicated by a plus sign). Human vision perceives the result as 'Look like no change' (with a blue checkmark), while machine vision perceives it as 'Different identity' (with a red checkmark).

Fig. 4. Intuitive effect of perturbation-based methods.

Adversarial perturbations are crafted quasi-imperceptible noises that are added to the input data, causing deep learning models to make incorrect prediction. Because of the slight impact on human visual perception, adversarial perturbations can be applied in subject face privacy protection, called *perturbation-based methods*.

Given a face image  $x$ , the crafted adversarial perturbation  $\Delta x$  can prevent the automatic identity extraction from the

{4}------------------------------------------------

unauthorized FR model  $f(\cdot)$ . Perturbation-based methods can be formalized to solve a constrained optimization problem:

$$\begin{aligned} \max_{\Delta x} \mathcal{D}(f(x + \Delta x), f(x)), \\ \text{s.t. } \|\Delta x\|_p \leq \epsilon, \end{aligned} \quad (1)$$

where  $\mathcal{D}(\cdot, \cdot)$  is a metric of identity distance (typically Euclidean or cosine distance),  $\|\cdot\|_p$  denotes the  $l_p$  norm (typically  $p = 1, 2, \infty$ ), and  $\epsilon$  constrains the maximum extent of the perturbation deviation. In Equation 1, maximizing the objective function can achieve **Goal ②**, and the constraint term can promise **Goal ①**.

Through solving the above optimization problem,  $\Delta x$  that satisfies the requirements can be constructed to protect face privacy. As shown in Fig. 4, the protected image is only added with a small level of noise, making human vision think that the image looks the same before and after the protection. Therefore, familiar persons can still recognize the subject identity (**Goal ①**). In addition, machine vision is disturbed by perturbations and recognizes the face in the protected image as others. Therefore, FR tools cannot extract facial identity, which achieves (**Goal ②**).

### B. A False Sense of Privacy

Thanks to the almost unchanged visual content, perturbation-based methods can easily maintain identity consistency for human vision. However, we point out that it just provides a false sense of privacy, which cannot stand up to real-world challenges.

- 1) **Inherent Unsustainability:** Perturbation-based methods rely on exploiting the vulnerabilities of deep neural networks to deceive FR systems [19]. However, these vulnerabilities are not static; modern FR systems are continuously updated and retrained to fix known weaknesses. As a result, adversarial perturbation that once succeeded may lose their effectiveness as the systems evolve. This leads to a persistent *cat-and-mouse game*—as new perturbations are created, new defenses emerge, rendering past efforts obsolete. Thus, such methods fail to provide sustainable protection for users.
- 2) **Unattainable Transferability:** It is unattainable to craft a universal adversarial perturbation that can counteract all FR systems. Different FR systems have differences in architectural design, training strategy, data distribution, or optimization. These factors result in different decision boundaries and vulnerabilities. While some studies attempt to improve transferability [51], their success is often limited to specific model families. Thus, such methods fail to provide transferable protection for users.
- 3) **Weak Robustness:** It is difficult for adversarial perturbations to survive in the OSN transmission. OSNs perform various image processing on photos shared by users to improve loading speed, save bandwidth, adapt to device screens, etc. Common operations like JPEG compression would alter the image pixels, which may remove the adversarial perturbations. While the employment of noise layers [52] improves robustness to noise, it is impossible

to enumerate all the noises. Thus, such methods fail to provide robust protection for users.

- 4) **Wrong Priority:** As defined in Equation 1, the constraint is designed for utility, which gives utility a higher priority than privacy. That is, privacy is not necessarily satisfied in the optimized result. We argue that this prioritization is flawed: the risk of facial identity leakage typically outweighs the minor loss in visual fidelity. Therefore, the optimization problem Equation 1 should be reset to privacy as the constraint and utility as the optimization objective. Thus, such methods fail to provide prior protection for users.

### C. Evaluation of Reliable Protection

In the previous subsection, we analyzed the limitations of perturbation-based methods in protection from four aspects. This section verifies the above conclusion through experiments. Specifically, we select three representative methods for evaluation, including:

- **Fawkes [17]:** Fawkes is a popular perturbation-based protection system, which can generate quasi-imperceptible noise for individual images. The team of Fawkes also provides applications for Windows and MacOS, which drive the progress of privacy protection for everyone. Fawkes contains three levels of protection to trade off privacy and utility. To adapt to real-world application scenarios, we use the middle level of protection for evaluation.
- **OPOM [28]:** Unlike image-specific Fawkes, OPOM constructs user-specific perturbations by exploiting the identity space of a single user, which can protect all the face images of that user. In addition, it uses momentum enhancement methods and DFANet to improve the transferability. Based on the recommendation in the paper, we set  $\epsilon$  to 8 to carry out the evaluation.
- **AMT-GAN [18]:** Unlike Fawkes and OPOM which both use *pixel-level noise* as perturbation, AMT-GAN employs *semantic-level noise* (makeup) to enhance visual naturalness. In addition, it employs an ensemble training strategy with input diversity enhancement to enhance the transferability and robustness. We carried out the evaluation based on the pre-trained model provided by the authors.

There are also some related methods that were not selected for testing because of similar design ideas or non-open-source codes, while the chosen methods were sufficiently representative of them.

We conducted two types of evaluations, including adding noise (for robustness) and utilizing commercial APIs (for transferability and sustainability). Please note that priority is intuitive and thus doesn't require to be evaluated

**Against Noise.** To reduce the bandwidth requirements for storage and transmission, uploaded photos are typically JPEG compressed. Meanwhile, during transmissions, random noise may be introduced, which often exhibits Gaussian distribution. To evaluate the *robustness*, we apply JPEG compression ( $Q = 90$ ) and Gaussian noise ( $\sigma = 0.1$ ).

Specifically, we selected FaceNet [53] and IR152 [54] to evaluate identity change before and after protection. To

{5}------------------------------------------------

![Figure 5: Perturbation-based against noise. Two box-and-line plots showing cosine similarity for FaceNet and IR152 models. The x-axis for both models shows three methods: Fawkes, OPOM, and AMT-GAN, each with two noise types: -J (JPEG compression) and -G (Gaussian noise). The y-axis represents Similarity from 0.0 to 0.8. For FaceNet, Fawkes and OPOM show higher similarity with -J noise compared to -G noise, while AMT-GAN shows more consistent performance. For IR152, all methods show lower similarity with -J noise compared to -G noise.](55d2bfe1c3d04e86df8d7a104d802172_img.jpg)

Figure 5: Perturbation-based against noise. Two box-and-line plots showing cosine similarity for FaceNet and IR152 models. The x-axis for both models shows three methods: Fawkes, OPOM, and AMT-GAN, each with two noise types: -J (JPEG compression) and -G (Gaussian noise). The y-axis represents Similarity from 0.0 to 0.8. For FaceNet, Fawkes and OPOM show higher similarity with -J noise compared to -G noise, while AMT-GAN shows more consistent performance. For IR152, all methods show lower similarity with -J noise compared to -G noise.

Fig. 5. Perturbation-based against noise, where “-J” means JPEG compression and “-G” means Gaussian noise.

![Figure 6: Perturbation-based methods against commercial APIs. Two box plots showing confidence (%) for Amazon and Face++ APIs. The x-axis shows three methods: Fawkes, OPOM, and AMT-GAN. The y-axis represents Confidence (%) from 50 to 100. A red dashed line indicates the threshold at 80.00% for Amazon and 74.00% for Face++. For Amazon, all methods are above the threshold, with AMT-GAN showing the highest confidence. For Face++, all methods are also above the threshold, with OPOM showing the highest confidence.](ac99eff233b8fe51d30f499e7413c345_img.jpg)

Figure 6: Perturbation-based methods against commercial APIs. Two box plots showing confidence (%) for Amazon and Face++ APIs. The x-axis shows three methods: Fawkes, OPOM, and AMT-GAN. The y-axis represents Confidence (%) from 50 to 100. A red dashed line indicates the threshold at 80.00% for Amazon and 74.00% for Face++. For Amazon, all methods are above the threshold, with AMT-GAN showing the highest confidence. For Face++, all methods are also above the threshold, with OPOM showing the highest confidence.

Fig. 6. Perturbation-based methods against commercial APIs. Most of the results are above the thresholds (red line).

demonstrate the effect of noise, we selected 100 test samples where the perturbation had a large change in identity. Fig. 5 shows box-and-line plots of cosine similarity. Fawkes and OPOM are obviously affected by JPEG compression and Gaussian noise, resulting in a degradation of protection performance. This is because the perturbations generated by them are inherently pixel-level noise, which is very easily weakened or corrupted by pixel changes caused by JPEG compression and Gaussian noise. AMT-GAN utilizes makeup as a natural perturbation, which avoids the effect of JPEG compression but still brings a weak effect from Gaussian noise. Therefore, the reliable protection of perturbation-based methods is insufficient when faced with noise.

**Against Commercial APIs.** Advanced FR APIs do not disclose their models and parameters, making it difficult for existing techniques to consider them as surrogate models for perturbation generation. Therefore, these APIs can be well used to evaluate the *transferability* of privacy protection in real scenarios. In particular, commercial APIs periodically patch existing vulnerabilities to improve identification performance, which can help us evaluate the *sustainability* of privacy protection.

Specifically, we selected Amazon and Face++ to evaluate identity change before and after protection, where the matching threshold of Amazon is 80.00% and Face++ is 74.00%. As shown in Fig. 6, we can find that OPOM has the worst identity protection. Fawkes will also update its systems to improve transferability, but brings only a weak effect. AMT-GAN further improves the transferability through integration strategies, which provides the best protection. However, most of the results are above the thresholds, allowing commercial APIs to extract real identity features as well. Although these methods were effective in resisting commercial APIs at the

![Figure 7: Other perturbation-based methods against Face++. A grid of 12 pairs of face images showing the results of different perturbation methods. Each pair includes a visual comparison of the original and perturbed faces, a numerical confidence score, and a list of thresholds. The methods shown are: TIP-IM (2021ICCV), Adv-Attribute (2022NeurIPS), CLIP2Protect (2023CVPR), ImU (2023S&P), UnGANable (2023USENIX Security), GIFT (2024ACM MM), P3-Mask (2024ECCV), DiffAM (2024CVPR), DiffPrivate (2025PETS), and DynTracker (2025CCS). All confidence scores are above the 74% threshold.](130ce5fbe21d11939cd4419d3e7ff044_img.jpg)

Figure 7: Other perturbation-based methods against Face++. A grid of 12 pairs of face images showing the results of different perturbation methods. Each pair includes a visual comparison of the original and perturbed faces, a numerical confidence score, and a list of thresholds. The methods shown are: TIP-IM (2021ICCV), Adv-Attribute (2022NeurIPS), CLIP2Protect (2023CVPR), ImU (2023S&P), UnGANable (2023USENIX Security), GIFT (2024ACM MM), P3-Mask (2024ECCV), DiffAM (2024CVPR), DiffPrivate (2025PETS), and DynTracker (2025CCS). All confidence scores are above the 74% threshold.

Fig. 7. Other perturbation-based methods against Face++. All of the results were above the threshold (74%) and thus privacy is not protected.

time in the original papers, their protection performance has significantly declined with updates and patches to the commercial APIs, rendering them only temporarily effective.

In addition, we cropped images from other similar methods for testing, e.g., TIP-IM [15], ImU [33], CLIP2Protect [55], and GIFT [34]. In Fig. 7, the results show that these methods still keep a high confidence of face matching, thus enabling to extract personal identity. Therefore, due to periodic vulnerability fixes, the protection is no longer reliable against the ever-evolving commercial APIs.

## V. KEY THOUGHT AND INSIGHT

### A. Is It Feasible for Synthesis-based Methods to Protect Subject Face Privacy?

Previous section reveals that adversarial perturbation cannot provide reliable privacy protection. It is worth exploring whether synthesis-based methods can be an alternative solution to protect subject face privacy.

**Advantage.** Synthesis-based methods can achieve reliable privacy protection for **Goal ②**.

- **Sustainability:** They do not depend on vulnerabilities in FR systems, so identity protection still works even if the vulnerabilities are fixed.
- **Transferability:** The goal of all FR systems is to extract the identity in a face, so they are all capable of extracting the changed identity.
- **Robustness:** As a high-level semantic [54], [56], facial identity can show excellent robustness, so the extraction of the changed identity will not be significantly affected by common noises.

{6}------------------------------------------------

- *Priority*: They often minimize alterations to identity-unrelated attributes while changing identity, i.e., prioritizing privacy over utility.
- *Imperceptibility*: Compared to the irregular noise introduced by perturbation-based methods, synthesis-based methods provide distributions that are indistinguishable from the real data, which can be more imperceptible to the adversary.
- *Understandability*: Non-expert users can understand their privacy-preserving mechanism, i.e., changing identity, whereas users cannot understand how noise or make-up can attack FR systems.

**Challenge.** However, synthesis-based methods cannot achieve satisfactory utility for **Goal 1**. The main reason is that facial appearance is closely correlated with identity. The change in identity must modify facial appearance (e.g., skin color or eye shape), thus preventing familiar persons from recognizing subjects. It should be noted that all the existing synthesis-based methods [44], [57], [58] aim to anonymize visual appearance to block human vision. Although some works [22], [41] employ LPIPS to enhance visual similarity, the effect they bring is weak. Therefore, existing synthesis-based methods cannot be directly applied to subject face privacy protection, and it is challenging to achieve **Goal 1**.

**Thought.** Firstly, almost all works ignore a crucial point—Not “faces in the photo” but “the photo containing faces”. Existing methods deal with cropped faces locally, but neglect to consider the global impact on the photo after protection. In real scenarios, the main task of users is feeling the global message conveyed by the photo rather than the identity represented by the face. Therefore, a weak alteration of faces may not matter.

Secondly, the recognition by familiar persons is robust to image alterations and better than the recognition by unfamiliar persons [59], [60]. Perturbation-based methods bring subtle noise, which we consider to be *stringent* in real scenarios. In fact, even if the nose, eyes, and mouth of the face are altered *more relative to the noise*, familiar persons still can recognize the subject by viewing the photo.

Therefore, synthesis-based methods are *feasible to protect subject face privacy*, as long as the alteration of facial appearance is not sufficient to affect the recognition by human vision. For this, we should analyze how familiar persons recognize subjects in photos.

### B. How Familiar Persons Recognize Subjects in Photo Sharing Scenario?

Based on existing research and life experience, we present the following insight:

**Insight:** *In most photo sharing scenarios, the recognition of subjects by familiar persons relies on **identity perception** rather than **meticulous face analysis**.*

**Supporting Evidence.** We support the above insight from three perspectives.

- *From the perspective of photo content*, most shared photos don’t emphasize faces as the primary visual content (except

![Diagram illustrating identity perception. At the top, a photo of Hans Joas speaking is shown with a thought bubble saying 'Hans Joas is delivering a speech'. Below this, a stylized head icon represents the viewer's perception. The main part of the diagram shows two versions of the photo under the heading 'Identity Perception'. The first version, labeled '1 Contextual Perception of Non-Facial Regions', shows the photo with the face blurred (indicated by a red dashed line). The second version, labeled '2 Coarse-Grained Perception of Facial Regions', shows the photo with the face slightly blurred. Arrows indicate the flow from the original photo to these two perception models.](78d5774278a3f4a614f8c0ae485ce8d9_img.jpg)

Diagram illustrating identity perception. At the top, a photo of Hans Joas speaking is shown with a thought bubble saying 'Hans Joas is delivering a speech'. Below this, a stylized head icon represents the viewer's perception. The main part of the diagram shows two versions of the photo under the heading 'Identity Perception'. The first version, labeled '1 Contextual Perception of Non-Facial Regions', shows the photo with the face blurred (indicated by a red dashed line). The second version, labeled '2 Coarse-Grained Perception of Facial Regions', shows the photo with the face slightly blurred. Arrows indicate the flow from the original photo to these two perception models.

Fig. 8. A sample of identity perception, where the photo of Hans Joas is from IMDB-WIKI. The face region is blurred (red dashed line) to mimic coarse-grained perception.

for self-portraits) [61]. Instead, these photos typically highlight activities and events [62], e.g., playing sports, attending gatherings, or documenting daily routines, where the face may be small, partially visible, or even occluded. Therefore, the main task of friends is to interpret the social activity or event depicted in the photo, rather than to engage in face analysis, naturally facilitates identity perception by familiar persons.

- *From the perspective of cognitive*, familiar persons build stable mental representations of others through repeated social exposure. These representations integrate various personal characteristics [63] beyond facial geometry — such as clothing style, body posture, and even makeup. As such, familiar viewers can still reliably recognize subjects in degraded or partial-view conditions [64]. In addition, recognition by familiar persons is more robust to image degradation than that by unfamiliar persons [21], [65]. This also indicates that familiar persons do not rely on meticulous face analysis; otherwise, their recognition performance would be comparable to that of unfamiliar persons.
- *From the perspective of social context*, photo sharing often occurs within well-defined interpersonal boundaries—such as among friends or family—where the identity of the sharer and expected subjects are largely known. In these cases, recognition is more about confirming the presence of expected individuals than identifying unknown persons. Contextual factors [66], [67] like accompanying text, shared event themes, or the photo’s time of posting provide additional verification cues. The act of recognition in such contexts is often anticipatory and guided by shared background knowledge, which reduces the requirement for meticulous face analysis.

We define identity perception:

**Definition:** *Identity perception* refers to the quick and intuitive cognitive process by which familiar persons infer a person’s identity under non-strictly recognizable scenarios, combining contextual perception of non-facial regions and coarse-grained perception of facial regions, without meticulous face analysis.

{7}------------------------------------------------

![Diagram of the PerceptFace pipeline. It starts with an 'Original Photo X' which is AI-enlarged by 'Picsman'. Faces are then 'Detect & Crop Faces' to get 'Cropped Faces [x1, ..., xn]'. These are processed by 'PerceptFace', which has an 'Attribute Extractor' (outputting z_attr) and an 'Identity Extractor' (outputting z_id). These features go through a 'Transformer' (outputting z_id^t) and 'Feature Fusion' to a 'Generator' (outputting 'Protected Faces [x-hat1, ..., x-hatn]'). Finally, 'Reposition Faces' to get the 'Protected Photo X-hat'.](990567efebf979be51f56d1150012c9d_img.jpg)

Diagram of the PerceptFace pipeline. It starts with an 'Original Photo X' which is AI-enlarged by 'Picsman'. Faces are then 'Detect & Crop Faces' to get 'Cropped Faces [x1, ..., xn]'. These are processed by 'PerceptFace', which has an 'Attribute Extractor' (outputting z\_attr) and an 'Identity Extractor' (outputting z\_id). These features go through a 'Transformer' (outputting z\_id^t) and 'Feature Fusion' to a 'Generator' (outputting 'Protected Faces [x-hat1, ..., x-hatn]'). Finally, 'Reposition Faces' to get the 'Protected Photo X-hat'.

Fig. 9. The pipeline of PerceptFace, where the original photo is AI-enlarged by Picsman, avoiding potential ethical and copyright issue. Firstly, the subject faces are detected and cropped. Secondly, PerceptFace protects each subject face. Finally, the protected faces are repositioned to the original photo. Since PerceptFace only slightly modifies the facial area for *identity protection*, which allows familiar persons to *perceive identity* of the subject.

**Detail.** We detail the two sub-processes: 1) *Contextual perception of non-facial regions*. Apart from non-visual information (e.g., social context), appearance context (e.g., hairstyle, clothing, and physique) can stimulate the prior knowledge of friends and help them to quickly associate the closest identity. Especially for highly familiar subjects, even if their faces are not visible, friends can accurately recognize the subject through these appearance context. 2) *Coarse-grained perception of facial regions*. Coarse-grained facial characteristics (e.g., general contours, facial structure, and skin color) further help friends to convince the subject identity. In real-world scenarios, most friends just coarsely view the facial region without carefully comparing the facial details (e.g., precise nose size). Fig. 8 presents a sample process with *Hans Joas*.

## VI. METHODOLOGY

### A. Pipeline of PerceptFace

Based on the *insight* in the previous section, we make a modification to the design **Goal 1**. Then, our goals are:

**Goal 1\*:** *Preserving the utility that familiar persons can perceive identity of the subject via human vision.*

**Goal 2:** *Protecting the privacy that FR systems cannot extract identity of the subject via machine vision.*

To achieve the above goals, we propose the first synthesis-based method for subject face privacy *i.e.*, PerceptFace. Given a photo  $\mathcal{X}$ , the PerceptFace can edit the subject faces  $[x_1, \dots, x_n]$  in it, enabling the corresponding identity to be changed while maintaining a high level of visual perceptual similarity. When sharing the protected photo  $\hat{\mathcal{X}}$ , the sharer's friends can easily perceive the subject identity via human vision (**Goal 1\***), but the FR systems cannot extract accurately the identity via machine vision (**Goal 2**).

Fig. 9 shows the usage flow of the proposed PerceptFace in privacy-preserving social photo sharing.

**1) Firstly**, we detect and crop the subject faces in the photo. Since distinguishing between the subjects and bystanders is not the focus of our work, for the sake of simplicity, we assume that all persons in the photo are subjects.

$$[x_1, \dots, x_n] = \text{FaceDetect}(\mathcal{X}), \quad (2)$$

where  $\text{FaceDetect}(\cdot)$  is the face detector.

**2) Secondly**, for each subject face, the proposed PerceptFace is utilized to protect them. By elaborately changing the identity features, the protected face is only slightly different from the original face in visual perception, while it belongs to another identity in machine vision,

$$\hat{x}_i = \text{PerceptFace}(x_i), i = 1, \dots, n. \quad (3)$$

**3) Finally**, all of the protected faces are repositioned into the original photo, obtaining the protected photo  $\hat{\mathcal{X}}$ . Since the background areas of the protected faces remain undistorted, they are able to blend naturally into the original photo,

$$\hat{\mathcal{X}} = \text{FaceReposit}(\mathcal{X}, [\hat{x}_1, \dots, \hat{x}_n]). \quad (4)$$

For the protected photo, since the contextual information is fully preserved and the facial region is only slightly altered, the familiar persons can perceive the subject identity with their prior knowledge, thus preserving utility. Benefiting from the robustness of high-level semantic (identity) change, arbitrary FR systems can only extract the changed (not the original) identity in noisy environments, thus protecting privacy.

### B. PerceptFace

**Design Idea.** PerceptFace aims to make identity unextractable yet perceptible. As a synthesis-based method, PerceptFace is able to make identity not extractable, so the challenge lies in making identity perceptible. Following the process of identity perception mentioned in the previous section, our idea is to design:

- *For contextual perception of non-facial regions*, we disentangle identity and attribute features, which can reduce the changes of non-facial regions when protecting identity. (*Attribute-Preserved Identity Manipulation*)
- *For coarse-grained perception of facial regions*, we design an innovative perceptual similarity loss for faces, which can maintain high similarity perception of facial regions. (*Perception-Enhanced Identity Transformation*)

**Flow.** As shown in Fig. 9, PerceptFace firstly extracts the attribute features  $z_{attr}$  and identity features  $z_{id}$  of the cropped face  $x$  by the attribute extractor  $E_{attr}$  and identity extractor

{8}------------------------------------------------

$E_{id}$ , respectively. Secondly,  $z_{id}$  is elaborately modified via the designed transformer  $T$  to obtain the protected version  $z_{id}^t$ . Thirdly,  $z_{id}^t$  is fused with the original attribute  $z_{attr}$ , and then synthesized into the protected face  $\hat{x}$  by the generator. In this way, the protected face maintains a similar visual appearance with the original one, while the identity information is effectively protected. Note that *due to information loss during synthesis, the identity features  $\hat{z}_{id}$  of  $\hat{x}$  typically exhibit slight deviations from  $z_{id}^t$ .*

**Network Structure.** The *attribute extractor*  $E_{attr}$  adopts four convolutional layers for downsampling, followed by a batch normalization layer and a ReLU activation layer. We use the off-the-shelf pre-trained Arcface as the *identity encoder*  $E_{id}$ , which can accurately extract identity features. To fuse identity and attribute features, we introduce the ID Injection module [68] that can facilitate the generator to synthesize faces with the target identity and attributes. Similar to  $E_{attr}$ , the generator  $G$  adopts four deconvolutional layers for upsampling, followed by a batch normalization layer and a ReLU activation layer. The input and output of the *transformer*  $T$  are one-dimensional vectors of the same length, thus a simple two-layer perceptron can be its backbone. Lastly,  $L_2$  regularization is followed by the last layer to constrain identity features in hyperspherical spaces.

### C. Attribute-Preserved Identity Manipulation

Attribute-preserved identity manipulation (APIM) aims to preserve attributes while manipulating identity. Feature disentanglement can separate the face features into attribute features and identity features, which facilitates the preservation of appearance perception when manipulating identity. By changing only the identity features, face privacy can be protected while identity-irrelevant attributes are preserved, thus maintaining contextual information, e.g., background and hairstyle.

Since the identity extractor is pre-trained, it already has the ability to extract identity features accurately. Therefore, the remaining work is to train both the attribute extractor  $E_{attr}$  and the generator  $G$  to disentangle the attribute features and synthesize the specified face. Similar to the existing work [23], [68], we use the following losses:

**Identity Loss.** Identity loss forces the identity  $\hat{z}_{id}$  of the synthetic face to be consistent with the transformed identity  $z_{id}^t$ . Cosine similarity is used to measure identity loss:

$$\mathcal{L}_{id} = 1 - \frac{\hat{z}_{id} \cdot z_{id}^t}{\|\hat{z}_{id}\|_2 \|z_{id}^t\|_2}. \quad (5)$$

**Attribute Loss.** Attribute loss forces the attribute of the synthetic face to be consistent with the original face. Weak-feature matching loss [68] is used to measure attribute loss:

$$\mathcal{L}_{attr} = \sum_{i=h}^H \frac{1}{N_i} \|D_i(x) - D_i(\hat{x})\|_1, \quad (6)$$

where  $D_i(\cdot)$  represents the feature extracted from the  $i$ -th layer by the discriminator  $D$ ,  $N_i$  indicates the number of elements in the  $i$ -th layer,  $H$  denotes the total number of layers, and  $h$  specifies the starting layer for calculating the attribute loss.

**Fusion Loss.** The fusion loss can guide the attributes and modified identity to synthesize into the specified face. Since the specified face is not known in advance, the original face is usually used as the specified face. In other words, the original face is disentangled and then reconstructed back to the original face, which uses the unmodified identity features and attribute features. Specifically, the  $L_1$  loss is used to construct a pixel-level fusion loss.

$$\mathcal{L}_{fus} = \|G(z_{id}, z_{attr}) - x\|_1. \quad (7)$$

**Adversarial loss.** To improve visual quality, adversarial loss  $\mathcal{L}_{adv}$  [69] is used to make synthetic faces indistinguishable from real faces.

**Objective I.** Overall, the training objective of APIM is formulated as follows:

$$\mathcal{L}_{total}^I = \mathcal{L}_{adv} + \lambda_{id}\mathcal{L}_{id} + \lambda_{attr}\mathcal{L}_{attr} + \lambda_{fus}\mathcal{L}_{fus}, \quad (8)$$

where  $\lambda_*$  are hyperparameters which control the magnitude of different losses.

### D. Perception-Enhanced Identity Transformation

APIM in the previous subsection can preserve identity-irrelevant attributes (mainly non-facial regions), but this is not sufficient for maintaining identity perception consistency. As identity is likewise correlated with facial regions, arbitrary changes in identity can lead to distortions in the facial area, which can bring about obvious differences in visual perception, e.g., general contours, structure, and skin color. As shown in Fig. 10, it can be observed that the visual appearance of the protected face is significantly altered, which is adequate to affect the identity perception by familiar persons.

In this section, perception-enhanced identity transformation (PEIT) works on constructing an identity transformation to delicately edit the identity features, so as to maintain high visual perceptual similarity while changing the identity, which can be formalized as,

$$\begin{aligned} \max_{z_{id}^t} \mathcal{S}(G(z_{id}^t, z_{attr}), x), \\ \text{s.t. } \cos(\hat{z}_{id}, z_{id}) < \tau, \end{aligned} \quad (9)$$

where  $\mathcal{S}(\cdot, \cdot)$  is a metric of perceptual similarity,  $\cos(\cdot, \cdot)$  is the cosine similarity, and  $\tau$  is the threshold. Compared to Eq. 1, Eq. 9 treats privacy as a constraint to maximize the objective function (utility), which is more in line with the goal of privacy protection.

![Figure 10: A 2x6 grid of face images comparing 'Original' and 'Protected' results. The top row shows original faces of six different individuals. The bottom row shows the corresponding protected faces generated by APIM. The protected faces maintain the general appearance and facial features of the originals while altering the identity, demonstrating the effectiveness of the APIM method in preserving visual perception.](920019e1f338440f5bffe0f7574d91b9_img.jpg)

Figure 10: A 2x6 grid of face images comparing 'Original' and 'Protected' results. The top row shows original faces of six different individuals. The bottom row shows the corresponding protected faces generated by APIM. The protected faces maintain the general appearance and facial features of the originals while altering the identity, demonstrating the effectiveness of the APIM method in preserving visual perception.

Fig. 10. Protected results via APIM. Although the non-facial areas are preserved, the visual changes in the facial areas are still easily perceived.

{9}------------------------------------------------

**Feasibility Analysis:** Unless doing a facial analysis task, human vision usually does not pay more attention to fine-grained facial features. In the real world, we also often find examples of similar identities in human visual perception, especially between twins or between celebrities and specific public. Nevertheless, upon meticulous facial analysis, it can be distinguished that these perceptually similar identities are different, e.g., different sizes of noses or eye distances. Therefore, it is feasible to maintain high visual perceptual similarity while changing the identity.

To avoid the time-consuming problem caused by the iterative solution for single data, we train an MLP-based transformer  $T$  to provide real-time identity transformation, which is supervised by the following losses:

**Enhanced Face Perception for Utility.** To allow identity perception, the protected face should maintain a high visual perceptual similarity to the original face. However, it is difficult to directly measure visual face perception, which is a complex neural inference process.

As an alternative approach, learned perceptual image patch similarity (LPIPS) [70] is a perceptual similarity metric for generalized categories of images. Unlike the simple or low-level metrics (e.g., PSNR and SSIM), LPIPS utilizes the variability of depth features to measure perceptual similarity, which is more in alignment with human vision in texture, color, and structure. Therefore, we consider using LPIPS as the perceptual similarity loss.

$$\mathcal{L}_{lips} = \|\mathcal{P}(x) - \mathcal{P}(\hat{x})\|_2, \quad (10)$$

where  $\mathcal{P}(\cdot)$  is the pre-trained perceptual feature extractor, which is selected as AlexNet in our work. However, LPIPS targets generic category images, rather than face category images. Previous research [24] has shown that the human vision system appears to allocate specialized neural resources to face perception. Therefore, the ability of LPIPS to mimic then face perception is limited.

Existing research [71] pointed out that human vision has different perceptual sensitivity (PS) to different facial features via user study. Inspired by it, we design a new perceptual similarity loss for faces, which introduces PS to face regions based on LPIPS. Regions with higher PS would limit smaller changes.

Specifically, a pre-trained face parser (FP) is adopted to segment out face regions  $[M_1, \dots, M_k]$ , representing the masks of eyebrows, eyes, nose, mouth, and skin,

$$[M_1, \dots, M_k] = FP(x). \quad (11)$$

It should be noted that non-facial areas (e.g., hair) are not included, as these are preserved to a greater extent through APIM. Then, we calculate the corresponding region changes,

$$\mathcal{L}_{region} = \sum_{i=1}^k \alpha_i \|M_i \odot x - M_i \odot \hat{x}\|_2, \quad (12)$$

where the value of  $\alpha_i$  is taken as the maximum PS of the different features in the region in [71], as shown in Fig. 11. Specially,

$$\alpha_i = \frac{\max_{f \in \mathcal{F}_i} \text{PS}(f)}{\sum_j \max_{f \in \mathcal{F}_j} \text{PS}(f)}, \quad (13)$$

![Figure 11: Perceptual sensitivity (PS) of different facial regions. A diagram of a face with color-coded regions: eyebrows (dark blue), eyes (medium blue), nose (light blue), mouth (very light blue), and skin (white). A legend on the right lists the maximum PS for each region: eyebrows (0.63), eyes (0.73), nose (0.60), mouth (0.75), and skin (0.57). Darker colors indicate greater PS.](9551159385e06b56c90a1a26664533a2_img.jpg)

Figure 11: Perceptual sensitivity (PS) of different facial regions. A diagram of a face with color-coded regions: eyebrows (dark blue), eyes (medium blue), nose (light blue), mouth (very light blue), and skin (white). A legend on the right lists the maximum PS for each region: eyebrows (0.63), eyes (0.73), nose (0.60), mouth (0.75), and skin (0.57). Darker colors indicate greater PS.

Fig. 11. Perceptual sensitivity (PS) of different facial regions, where darker colors indicate greater PS.

where  $\mathcal{F}_i$  denotes the set of facial features in region  $M_i$ , and  $\text{PS}(f)$  is the PS score of feature  $f$  in  $M_i$ , as reported in [71]. For example, the eye region includes features such as eye color (PS = 0.73) and eye shape (PS = 0.61), and we choose the maximum value (0.73) as the representative PS for that region. Finally, the obtained values are normalized to produce the final perceptual sensitivity weights:  $\alpha = [0.192, 0.223, 0.183, 0.229, 0.174]$ .

Eventually, the designed perceptual similarity loss for faces is defined as,

$$\mathcal{L}_{per} = \mathcal{L}_{lips} + \mathcal{L}_{region}. \quad (14)$$

**Basic Identity Deviation for Privacy.** To ensure identity protection, the faces to be protected should be moved away from the original face in the identity feature space, effectively minimizing their similarity. Specifically, when the similarity drops below a predefined threshold  $\epsilon$ , no further reduction in loss is necessary. Therefore, the identity deviation loss can be expressed as:

$$\mathcal{L}_{pri} = \max(\epsilon, \cos(E_{id}(x), E_{id}(\hat{x}))). \quad (15)$$

Only a single identity extractor is employed, which is sufficient to change identity. Of course, integrating results from multiple identity extractors is also recommended to avoid model bias, but not necessary.

**Objective II.** Overall, the training objective of PEIT is formulated as follows:

$$\mathcal{L}_{total}^{II} = \lambda_{pri} \mathcal{L}_{pri} + \lambda_{per} \mathcal{L}_{per}, \quad (16)$$

where  $\lambda_*$  are hyperparameters which control the magnitude of different losses.

## VII. EXPERIMENTAL RESULTS

### A. Setup

1) **Datasets:** VGGFace2 is a large-scale face dataset comprising over 3.31 million images of 9,131 identities, featuring significant variations in pose, age, illumination, ethnicity, and profession. All images are resized to  $224 \times 224$ . We use the face images of 8,631 identities for training, while the remaining identities are reserved for testing. CelebA-HQ, widely utilized for face recognition tasks, includes 30,000 aligned face images at a resolution of  $1024 \times 1024$ . Each image is annotated with 5 landmarks and 40 binary attributes. Similarly, we resize these images to  $224 \times 224$ . To assess the generalizability of our method to other datasets, CelebA-HQ is exclusively used for testing.

Since the social photo sharing scenario involves sharing photos that contain faces (rather than cropped faces), we use uncropped photos (IMDB-WIKI) for the rest of the experiment.

{10}------------------------------------------------

![Figure 12: Visual results of our method and baselines. The figure is a 4x6 grid of face images. The columns are labeled 'Original', 'RiDDLE', 'Disguise', 'Fawkes', 'AMT-GAN', and 'Ours'. The rows show different faces. Below the grid, 'Synthesis-based' is labeled under RiDDLE and Disguise, and 'Perturbation-based' is labeled under Fawkes and AMT-GAN. 'Ours' is not categorized.](6d9013c24741e861f3c8e0a763b6da22_img.jpg)

Figure 12: Visual results of our method and baselines. The figure is a 4x6 grid of face images. The columns are labeled 'Original', 'RiDDLE', 'Disguise', 'Fawkes', 'AMT-GAN', and 'Ours'. The rows show different faces. Below the grid, 'Synthesis-based' is labeled under RiDDLE and Disguise, and 'Perturbation-based' is labeled under Fawkes and AMT-GAN. 'Ours' is not categorized.

Fig. 12. Visual results of our method and baselines. Compared to synthesis-based methods, ours has higher perceptual similarity. Compared to perturbation-based methods, ours has higher visual naturalness.

2) *Dual-phase Training Strategy*: To avoid conflicts caused by multi-objective optimization, we adopt a two-stage training strategy to learn the parameters of PerceptFace. 1) *In the first stage*, we utilize *Objective I* to train  $E_{attr}$  and  $G$  to achieve attribute-preserved identity manipulation. Specifically, both  $E_{attr}$  and  $G$  are optimized with Adam optimizer with  $(\beta_1 = 0.5, \beta_2 = 0.99)$ , the initial learning rate is set to 0.0004, and the hyperparameters are set to  $\lambda_{id} = 30$ ,  $\lambda_{attr} = 10$ , and  $\lambda_{fus} = 10$ . 2) *In the second stage*, we utilize *Objective II* to train  $T$  to achieve perception-enhanced identity transformation. Specifically,  $T$  is optimized with Adam optimizer with  $(\beta_1 = 0.99, \beta_2 = 0.99)$ , the initial learning rate is set to 0.0004, and the hyperparameters are set to  $\lambda_{pri} = 5$  and  $\lambda_{per} = 5$ .

3) *Baselines*: We selected representative synthesis-based methods, i.e., RiDDLE [22] (entirely new face) and Disguise [23] (only altered in identity). RiDDLE is a de-identification model that synthesizes the entire head, which also synthesizes non-facial regions (e.g., hair) to provide enhanced visual anonymity. It also can restore the original face through the correct password. Disguise employs feature disentanglement to independently change identity to preserve more attributes, which is more consistent with our work. In addition, we also selected the representative perturbation-based methods, Fawkes [17] (pixel-level) and AMT-GAN [18] (semantic-level), to show the protection capability of our method.

### B. Evaluation of Privacy Protection

The privacy protection goal is to prevent the FR systems from extracting the original identity of the face. Fig. 12 illustrates the protection results of our method and baselines. To evaluate privacy, we utilize FR tools to extract the identity features of the faces before and after protection, respectively, and then compare their similarity. Advanced FR models and APIs are adopted as the tools, including IR152, IRSE50,

![Figure 13: Some test results in Face++, presenting the effectiveness and practicality of our method. The figure shows two sets of face images. The left set shows a woman's face with bounding boxes and JSON data. The right set shows a man's face with bounding boxes and JSON data. The JSON data includes fields like 'request_id', 'confidence', 'time_used', 'threshold', and 'thresholds'.](8c49beafe7c55ea578abcaee4bdca0ba_img.jpg)

Figure 13: Some test results in Face++, presenting the effectiveness and practicality of our method. The figure shows two sets of face images. The left set shows a woman's face with bounding boxes and JSON data. The right set shows a man's face with bounding boxes and JSON data. The JSON data includes fields like 'request\_id', 'confidence', 'time\_used', 'threshold', and 'thresholds'.

Fig. 13. Some test results in Face++, presenting the effectiveness and practicality of our method.

FaceNet, MobileFace, Amazon<sup>1</sup>, and Face++<sup>2</sup>, where the models adopt cosine similarity and the APIs use the self-defined similarity.

1) *PerceptFace exhibits significantly stronger privacy protection performance than perturbation-based methods, and comparable performance compared to existing synthesis-based methods.*

**Low Identity Similarity.** Table I shows the identity similarity before and after protection. Figure 13 shows examples of PerceptFace by Face++. The top three best performances are labeled blue, and darker colors indicate better protection. Disguise is the most effective because of the significant modifications made to the identity features. RiDDLE, although it generates a completely new head, brings about a relatively small change in identity compared to Disguise because of the need to support reversibility. Compared to these synthetic-based methods, our method has a relatively weak identity deviation capability, but also significantly reduces identity similarity. In particular, our method also outperforms RiDDLE in some of the results. These perturbation-based methods also have the ability to reduce the identity similarity but this is too weak compared to our method.

**High PSRs.** In Table II, we further calculated the protection success rate (PSR) based on the matching thresholds at 0.001 FAR for all FR models. Consistent with the results analyzed in Table I, the best performance is still achieved by these synthesis-based methods. Our method also achieves a high PSR of more than 90%, which is much higher than that of the perturbation-based method. Perturbation-based methods achieve relatively high PSRs only on specific FR tools but extremely low results on others, which also reveals that these methods have poor transferability. Furthermore, the protection performance of our method drops a bit on IRSE50 and MobileFace. This is due to the fact that their model architectures are smaller, making them also focus on coarse-grained face perception to some extent. While ever-evolving FR tools will use more refined networks and thus focus on fine-grained facial analysis, e.g., Face++. Therefore, the high accuracy of our method on APIs indicates the effectiveness of its protection.

It is worth emphasizing that in real world scenarios, the protected image obtained by an adversary is never the same as the unprotected face it holds. The protected image and unprotected image are just different face images belonging

<sup>1</sup><https://aws.amazon.com/cn/rekognition/>

<sup>2</sup><https://www.faceplusplus.com.cn/face-comparing/>

{11}------------------------------------------------

TABLE I  
IDENTITY SIMILARITY BEFORE AND AFTER PROTECTION UNDER DIFFERENT FR TOOLS.

|        |            | VGGFace2        |          |                    |         |        | CelebA-HQ       |          |                    |         |        |
|--------|------------|-----------------|----------|--------------------|---------|--------|-----------------|----------|--------------------|---------|--------|
|        |            | Synthesis-based |          | Perturbation-based |         |        | Synthesis-based |          | Perturbation-based |         |        |
|        |            | RiDDLE          | Disguise | Fawkes             | AMT-GAN | Ours   | RiDDLE          | Disguise | Fawkes             | AMT-GAN | Ours   |
| Models | FaceNet    | 0.1029          | -0.0171  | 0.5495             | 0.6746  | 0.3729 | 0.0758          | -0.0396  | 0.5497             | 0.5367  | 0.3063 |
|        | IR152      | 0.0291          | -0.0692  | 0.4420             | 0.5238  | 0.0288 | 0.0915          | -0.0841  | 0.4450             | 0.3586  | 0.0209 |
|        | IRSE50     | 0.0714          | -0.0961  | 0.6728             | 0.6776  | 0.1179 | 0.1889          | -0.0824  | 0.6688             | 0.5739  | 0.1414 |
|        | MobileFace | 0.1292          | -0.0396  | 0.6888             | 0.6767  | 0.2269 | 0.2199          | 0.0171   | 0.6835             | 0.6175  | 0.2838 |
| APIs   | Amazon     | 0.1252          | 0.0936   | 0.8543             | 0.8540  | 0.3858 | 0.1340          | 0.0669   | 0.8764             | 0.7719  | 0.3421 |
|        | Face++     | 0.3057          | 0.2592   | 0.8942             | 0.8641  | 0.5319 | 0.4311          | 0.2861   | 0.8834             | 0.8038  | 0.5267 |

TABLE II  
PROTECTION SUCCESS RATES UNDER DIFFERENT FR TOOLS.

|        |            | VGGFace2        |          |                    |         |         | CelebA-HQ       |          |                    |         |         |
|--------|------------|-----------------|----------|--------------------|---------|---------|-----------------|----------|--------------------|---------|---------|
|        |            | Synthesis-based |          | Perturbation-based |         |         | Synthesis-based |          | Perturbation-based |         |         |
|        |            | RiDDLE          | Disguise | Fawkes             | AMT-GAN | Ours    | RiDDLE          | Disguise | Fawkes             | AMT-GAN | Ours    |
| Models | FaceNet    | 99.40%          | 100.00%  | 58.20%             | 26.20%  | 93.70%  | 99.90%          | 100.0%   | 54.00%             | 62.20%  | 96.60%  |
|        | IR152      | 99.80%          | 99.90%   | 8.00%              | 3.40%   | 98.90%  | 92.70%          | 99.60%   | 7.60%              | 19.60%  | 98.40%  |
|        | IRSE50     | 98.60%          | 99.90%   | 0.40%              | 0.20%   | 96.90%  | 95.70%          | 99.30%   | 0.50%              | 1.70%   | 93.00%  |
|        | MobileFace | 98.70%          | 99.70%   | 0.30%              | 1.00%   | 92.60%  | 93.20%          | 98.80%   | 0.20%              | 0.50%   | 90.70%  |
| APIs   | Amazon     | 100.00%         | 100.00%  | 19.00%             | 25.00%  | 100.00% | 100.00%         | 100.00%  | 17.00%             | 41.00%  | 100.00% |
|        | Face++     | 100.00%         | 100.00%  | 2.00%              | 5.00%   | 99.00%  | 100.00%         | 100.00%  | 3.00%              | 7.00%   | 99.50%  |

![Figure 14: Identity similarity against JPEG -J(Q) and Gaussian -G(σ). The figure contains two box plots. The left plot is for FaceNet and the right plot is for IR152. Both plots show 'Cosine Similarity' on the y-axis (ranging from -0.2 to 0.8) against different noise levels on the x-axis: Protected, -J(90), -J(80), -J(70), -G(0.1), -G(0.15), -G(0.2). The 'Protected' box is red, and the others are blue. The similarity remains high and stable across all noise levels for both models.](dd330f8b8f6c16eae20c3a676b4eb804_img.jpg)

Figure 14: Identity similarity against JPEG -J(Q) and Gaussian -G(σ). The figure contains two box plots. The left plot is for FaceNet and the right plot is for IR152. Both plots show 'Cosine Similarity' on the y-axis (ranging from -0.2 to 0.8) against different noise levels on the x-axis: Protected, -J(90), -J(80), -J(70), -G(0.1), -G(0.15), -G(0.2). The 'Protected' box is red, and the others are blue. The similarity remains high and stable across all noise levels for both models.

Fig. 14. Identity similarity against JPEG  $-J(Q)$  and Gaussian  $-G(\sigma)$ . Our method is robust enough against noise.

TABLE III  
PSRS AGAINST OSNs IN IMDB-WIKI.

|        | Facebook | Instagram | WeChat | QQ   | Micro-blog |
|--------|----------|-----------|--------|------|------------|
| Amazon | 100.00%  | 100.00%   | 100%   | 100% | 100%       |
| Face++ | 100.00%  | 100.00%   | 100%   | 100% | 100%       |

to the same identity, which would further reduce the identity similarity and increase PSR.

2) *PerceptFace exhibits satisfactory robustness to common noise and online social networks (OSNs).*

**Against Noise.** Fig. 14 illustrates the similarity change under JPEG compression and Gaussian noise processing. We set the quality factor for JPEG compression to  $Q = 70, 80, 90$  and the standard deviation of Gaussian noise to  $\sigma = 0.1, 0.15, 0.2$ . It can be observed that the noise introduced by these operations does not significantly change the identity similarity of the protected results, making the impact on privacy protection weak. Therefore, PerceptFace is robust to common noise.

![Figure 15: Examples with different sizes in IMDB-WIKI, simulating real photos instead of just human faces. The figure is divided into 'Original' and 'Protected' columns. Each column shows two rows of images. The top row shows a person in a blue jersey and yellow shorts, and the bottom row shows a person in a dark jacket and light pants. The 'Protected' column shows the same images but with different sizes and aspect ratios, demonstrating the method's ability to handle non-standard image sizes.](077e3e6611a6d0ef38285c262298f237_img.jpg)

Figure 15: Examples with different sizes in IMDB-WIKI, simulating real photos instead of just human faces. The figure is divided into 'Original' and 'Protected' columns. Each column shows two rows of images. The top row shows a person in a blue jersey and yellow shorts, and the bottom row shows a person in a dark jacket and light pants. The 'Protected' column shows the same images but with different sizes and aspect ratios, demonstrating the method's ability to handle non-standard image sizes.

Fig. 15. Examples with different sizes in IMDB-WIKI, simulating real photos instead of just human faces.

**Against OSNs.** In the real world, users often upload images of different sizes instead of the cropped fixed-size face images. For this reason, we select uncropped images for testing at IMDB-WIKI, where images containing the full body of the subject are more likely to be selected. Considering that the image processing in OSNs is not publicly available, we selected only 20 images to be manually uploaded and downloaded in OSNs, including Facebook, Instagram, WeChat, QQ, and Micro-blog. Fig. 15 shows the example results. We directly use face recognition APIs to detect faces and compare their identity similarity. As shown in Table III, after the above OSNs processing, our method is still effective in protecting the identity. Therefore, PerceptFace is robust to OSNs.

Similar to RiDDLE and Disguise, PerceptFace makes

{12}------------------------------------------------

![Figure 16: Pixel-level and local region visual differences of our method. The figure is a grid of images. The first three columns show 'Original' (top row), 'Protected' (middle row), and 'Difference' (bottom row) for six different faces. The last four columns show zoomed-in views of the 'Difference' images for the same six faces, focusing on the Eyebrow, Eye, Nose, and Mouth regions. The 'Difference' images show very subtle changes, indicating high identity preservation.](bb6d33498937738ff5dac8d91c9ebaad_img.jpg)

Figure 16: Pixel-level and local region visual differences of our method. The figure is a grid of images. The first three columns show 'Original' (top row), 'Protected' (middle row), and 'Difference' (bottom row) for six different faces. The last four columns show zoomed-in views of the 'Difference' images for the same six faces, focusing on the Eyebrow, Eye, Nose, and Mouth regions. The 'Difference' images show very subtle changes, indicating high identity preservation.

Fig. 16. Pixel-level and local region (the last four columns) visual differences of our method.

TABLE IV  
IMAGE SIMILARITY OF DIFFERENT-LEVEL METRICS.

|                  |        | VGGFace2        |          |                    |         |        | CelebA-HQ       |          |                    |         |        |
|------------------|--------|-----------------|----------|--------------------|---------|--------|-----------------|----------|--------------------|---------|--------|
|                  |        | Synthesis-based |          | Perturbation-based |         |        | Synthesis-based |          | Perturbation-based |         |        |
|                  |        | RiDDLE          | Disguise | Fawkes             | AMT-GAN | Ours   | RiDDLE          | Disguise | Fawkes             | AMT-GAN | Ours   |
| Perceptual Level | LPIPS↓ | 0.343           | 0.100    | 0.033              | 0.114   | 0.059  | 0.283           | 0.087    | 0.023              | 0.088   | 0.058  |
| Structural level | SSIM↑  | 0.499           | 0.737    | 0.976              | 0.850   | 0.826  | 0.493           | 0.764    | 0.980              | 0.863   | 0.842  |
| Pixel level      | L1↓    | 0.102           | 0.060    | 0.009              | 0.081   | 0.032  | 0.097           | 0.055    | 0.008              | 0.069   | 0.031  |
| Pixel level      | RMSE↓  | 0.142           | 0.089    | 0.016              | 0.103   | 0.048  | 0.136           | 0.082    | 0.015              | 0.091   | 0.047  |
| SNR level        | PSNR↑  | 7.096           | 21.179   | 35.818             | 20.150  | 26.562 | 17.420          | 21.857   | 36.813             | 21.197  | 26.664 |

change directly to the high-level semantics (identity). Benefiting from semantic robustness [54], the altered identity of the face in the protected image remains stably extracted under the influence of various noises, thus preventing the leakage of the original identity.

### C. Evaluation of Utility Preservation

PerceptFace applies subtle alteration to only facial regions, which doesn't affect the utility of photos (except facial identity). Therefore, the evaluation of utility preservation goal is only for identity perception. Since human perception cannot be measured quantitatively, we use image similarity to approximate it. In addition, we also conduct user study for evaluating identity perception, which was done ethically.

**Ethical Considerations.** All participants provided informed consent for their data to be used in this research. Given the minimal risk to individuals, the public nature of the data, and the absence of sensitive information processing, our institutions consider this study exempt from IRB review under relevant institutional guidelines for low-risk research.

1) *PerceptFace exhibits significantly stronger image similarity performance than synthesis-based methods and AMT-GAN (semantic level), and relatively low performance compared to Fawkes (pixel level).*

**High Image Similarity.** Table IV shows the results of various image similarity metrics. Based on different similarity levels, we chose a variety of metrics, including perceptual level (LPIPS), structural level (SSIM), pixel level (L1, RMSE), and signal-to-noise ratio (SNR) level (PSNR). Fawkes achieves

the best image similarity owing to the small addition of pixel-level noise, thus retaining the most utility. AMT-GAN achieves perturbation by changing the semantics (make-up), which alters image content to a greater extent. Thus, similar to Disguise, which only changes facial regions, AMT-GAN only maintains a low image similarity. Our method further reduces the alteration of facial regions on the basis of Disguise, which enhances image similarity to retain more utility. This is because the designed perceptual loss limits the different variation of facial regions with different perceptual sensitivity.

**Low Visual Difference.** Fig. 12 shows a visual comparison of our method with baselines. It can be observed that the visual differences caused by our methods are imperceptible. Perturbation-based methods introduce visual unnaturalness, while synthesis-based methods have excessive visual content alterations. In Fig. 16, the results of the visual differences brought about by our method are further given. By careful observation, we can find weak facial contours, which can influence identity. Meanwhile, in Fig. 16, we carefully compare faces before and after protection, and it can be found that the nose, eyes and other identity-related areas undergone different degrees of slight changes. However, in social photo sharing scenarios, friends simply do not compare the original face to notice such weak alterations. Therefore, our method achieves a very low visual difference.

2) *PerceptFace maintains satisfactory identity perception and has an obvious preference in user study.*

**Satisfactory Identity Perception.** To simulate the prior knowledge of friends about users, we selected more famous celebrities for the evaluation, e.g., Cristiano Ronaldo. Specif-

{13}------------------------------------------------

![Figure 17: Ablation study results. The figure shows two rows of face images. The top row shows a man's face with yellow dotted boxes around the eyes and lower lip. The bottom row shows a man's face with yellow dotted boxes around the eyes and lower lip. The columns are labeled 'Original', 'W/o region', and 'Ours'. The 'W/o region' column shows noticeable changes in the eyes and lower lip, while the 'Ours' column shows imperceptible changes.](7832324609ad3cc688064e0341612b32_img.jpg)

Figure 17: Ablation study results. The figure shows two rows of face images. The top row shows a man's face with yellow dotted boxes around the eyes and lower lip. The bottom row shows a man's face with yellow dotted boxes around the eyes and lower lip. The columns are labeled 'Original', 'W/o region', and 'Ours'. The 'W/o region' column shows noticeable changes in the eyes and lower lip, while the 'Ours' column shows imperceptible changes.

Fig. 17. Ablation study, where “w/o region” has a noticeable change in the eyes and lower lip, while ours is imperceptible.

TABLE V  
USAGE PREFERENCE IN USER STUDY.

| RiDDLE | Disguise | Fawkes | AMT-GAN | No option | Ours          |
|--------|----------|--------|---------|-----------|---------------|
| 0.00%  | 0.00%    | 27.17% | 4.50%   | 3.83%     | <b>64.50%</b> |

ically, we carefully selected and downloaded some photos containing them for 10 celebrities in the online Google search, which was done ethically. Medium resolution, frontal face, normal expression, and inclusion of upper body were more likely to be selected. For each celebrity, we constructed 2 test samples. Each test sample contains 3 photos of the celebrity and 6 photos of others, of which only one photo of the celebrity is protected. The task of the participant is to find all the photos of that celebrity in these 9 photos. *As long as the protected photo is selected by the participant, the identity perception is successful.* If the participant does not know the celebrity, he or she has the option of skipping the test sample. We collected a total of 600 answers from 30 participants, among which 36 answers indicated that they did not know the celebrity, so 564 valid answers were obtained after elimination. Statistical results show that 513 out of these 564 valid answers were able to include the corresponding celebrity, so the identity perception rate reached 90.96%.

**Obvious Usage Preference.** We randomly selected 20 face images in CelebA-HQ and VGGFace2, where each image constitutes a test sample. Each test sample contains five protected faces (by baselines and our method, respectively), accompanied by a “no suitable option”. We collected a total of 600 answers from 30 participants, and calculated the proportion of each method chosen in Table V. No participants chose the synthetic-based methods because they significantly altered the perception of the correct identity by human vision. Only a small number of responses chose the AMT-GAN, because it brought too much flashy make-up in most cases, which is hard for users to accept. Fawkes is also chosen by some participants who are able to tolerate poor visualization from noise. Benefiting from favorable visualization and slight alterations, our method received 64.5% of choices, which indicates an obvious preference of our method.

TABLE VI  
IMAGE CHANGE DEGREE OF ABLATION STUDY.

|            | LPIPS↓          | Region_MSE↓     |
|------------|-----------------|-----------------|
| W/o region | 0.0548          | 0.1584          |
| Ours       | 0.0594(+0.0046) | 0.0948(-0.0636) |

### D. Ablation Study

To validate the effectiveness of the designed perceptual similarity loss, we remove the  $\mathcal{L}_{region}$  from this loss (w/o region), i.e., only LPIPS was used. **Qualitatively**, Fig. 17 presents the visual results, where the significantly changed areas are outlined by the yellow dotted lines. It can be observed that the eyes of the first face become noticeably larger after the protection of “w/o region”, while ours also changes the eyes but in a relatively weak way. The second face becomes thin in the lower lip after the protection of “w/o region”, while our change is imperceptible. As the findings of [71], human vision has higher sensitivity to eyes and lips, so our method can retain higher perceptual similarity than the “w/o region”. **Quantitatively**, we directly calculate the change in the facial region using  $\mathcal{L}_{region}$  and defined it as region\_MSE. Table VI shows the results. Ours is able to provide higher region\_MSE (0.0636) at the expense of almost negligible LPIPS (0.0046), reducing the alteration of highly sensitive regions.

### E. Limitation and Discussion

We believe PerceptFace has the potential to become a practical anti-FR tool widely used on real OSNs. Of course, some limitations are worth discussing.

1) *Applicable Photo Types:* PerceptFace is mainly applicable to photos that contain the upper body of the subject and show him/her participating in an activity or event, which helps friends to perceive identity through contextual information of non-facial areas, e.g., a photo of the subject playing soccer, attending a party with friends, or blending in with the natural environment. In contrast, the tool is not applicable to photos that aim to represent the subject, e.g., selfies or portraits, because these photos usually have facial details as the main visual focus and lack rich contextual features, making the identity perception process more difficult.

2) *Performance Degradation from Distribution Shift:* Faces in the real physical world have a different distribution than faces in the dataset, which degrades the performance of PerceptFace. Firstly, faces in datasets are aligned, whereas it is difficult to align and then protect faces in real-world photos. Secondly, faces in datasets are captured from early camera equipment, whereas the photos currently circulating on OSNs have higher resolution and more complex quality attributes with upgraded camera equipment. In the future, we consider leveraging domain adaptation, alignment-free modeling, and data augmentation techniques to address the performance degradation caused by distribution shift between dataset images and real-world photos.

3) *Identity Perception Quantization:* A user-friendly protection tool is suggested to provide quantifiable privacy and

{14}------------------------------------------------

utility. Privacy can be represented by the difference in identity feature similarity. However, the utility provided by PerceptFace remains difficult to quantify, i.e., what is the face perceptual similarity before and after protection? In this paper, we only provide an optimization loss to improve the face perceptual similarity, and the complexity of human vision still makes it difficult to present a quantitative metric.

4) *Security Issues from Limited Diversity*: We further assume that an adversary can collect a large number of paired protected and unprotected faces. They can invalidate our method by training an en-decoder network. Achieving diversity in PerceptFace can be an effective solution but is difficult. This is because identities with high perceptual similarity to a given face are rare. To improve security, we recommend that PerceptFace be securely managed and limited the frequency of its usage for a period of time.

5) *Failure Samples with Visible Nasal Distortion*: Unlike perturbation-based methods that iteratively optimize a single photo, PerceptFace uses neural networks to uniformly model and protect photos. While this approach possesses higher processing efficiency, it is difficult to fine-tune it for each photo, and thus the desired visual effect may not be obtained on some photos. As shown in Fig. 18, we observe that the nose appears significantly enlarged in some samples. This is due to the relatively low perceptual sensitivity of the nose region, and the model tends to apply a large distortion to it to achieve effective changes in identity information. This phenomenon reflects the essential difference in the objective function design between PerceptFace and perturbation methods, as detailed in Eq. 1 and Eq. 9. Nevertheless, PerceptFace enables users to intuitively perceive the degree of visual distortion, allowing them to make informed decisions about whether to share the photo. In contrast, although perturbation-based methods typically avoid noticeable distortion, their identity protection effectiveness is neither perceptible nor verifiable to users, making it difficult to establish sufficient trust.

![Figure 18: Failure samples with visible nasal distortion. The figure shows two rows of face images. Each row has two columns labeled 'Original' and 'Protected'. The first row shows a man's face, and the second row shows a woman's face. In the 'Protected' versions, the noses appear noticeably larger and more distorted compared to the 'Original' versions.](be217a121b8cc1b82eb1598749372865_img.jpg)

Figure 18: Failure samples with visible nasal distortion. The figure shows two rows of face images. Each row has two columns labeled 'Original' and 'Protected'. The first row shows a man's face, and the second row shows a woman's face. In the 'Protected' versions, the noses appear noticeably larger and more distorted compared to the 'Original' versions.

Fig. 18. Failure samples with visible nasal distortion.

## VIII. CONCLUSION

In this work, we explore a promising solution (synthesis-based methods) toward the development of more practical anti-FR tools for subject faces in photos. Firstly, we reveal that perturbation-based methods provide just a false sense of privacy. Secondly, we present an insight that in most photo sharing scenarios, the recognition of subjects relies on identity perception rather than meticulous face analysis by familiar persons. Finally, based on the insight, we propose a novel synthesis-based method for subject face privacy, i.e.,

PerceptFace, which renders identity unextractable yet perceptible. Meanwhile, we also design a new face perceptual similarity loss which introduces perceptual sensitivity based on LPIPS. Sufficient experiments show that PerceptFace achieves a satisfactory balance between utility and privacy, which is expected to advance the real-world application of face privacy protection technology.

## REFERENCES

- [1] A. D. Beldad and S. M. Hegner, "More photos from me to thee: Factors influencing the intention to continue sharing personal photos on an online social networking (osn) site among young adults in the netherlands," *Int. J. Hum-comput. Int.*, vol. 33, no. 5, pp. 410–422, 2017.
- [2] L. Laishram, M. Shaheryar, J. T. Lee, and S. K. Jung, "Toward a privacy-preserving face recognition system: A survey of leakages and solutions," *ACM Comput. Surv.*, 2024.
- [3] L. Verdoliva, "Media forensics and deepfakes: an overview," *IEEE J. Sel. Topics Signal Process.*, vol. 14, no. 5, pp. 910–932, 2020.
- [4] A. Morales, J. Fierrez, R. Vera-Rodriguez, and R. Tolosana, "SensitiveNets: Learning agnostic representations with application to face images," *IEEE Trans. Pattern Anal. Mach. Intell.*, vol. 43, no. 6, pp. 2158–2164, 2021.
- [5] J. Li, H. Zhang, S. Liang, P. Dai, and X. Cao, "Privacy-enhancing face obfuscation guided by semantic-aware attribution maps," *IEEE Trans. Inf. Forensics Security*, vol. 18, pp. 3632–3646, 2023.
- [6] E. Wenger, S. Shan, H. Zheng, and B. Y. Zhao, "Sok: Anti-facial recognition technology," in *Proc. IEEE Symp. Secur. Privacy*, 2023, pp. 864–881.
- [7] B. Meden, P. Rot, P. Terh rst, N. Damer, A. Kuijper, W. J. Scheirer, A. Ross, P. Peer, and V. Struc, "Privacy-enhancing face biometrics: A comprehensive survey," *IEEE Trans. Inf. Forensics Security*, vol. 16, pp. 4147–4183, 2021.
- [8] K. Tajik, A. Gunasekaran, R. Dutta, B. Ellis, R. B. Bobba, M. Rosulek, C. V. Wright, and W.-c. Feng, "Balancing image privacy and usability with thumbnail-preserving encryption," in *Proc. Netw. Distrib. Syst. Secur. Symp.*, 2019, p. 295.
- [9] L. Yuan, P. Korshunov, and T. Ebrahimi, "Privacy-preserving photo sharing based on a secure jpeg," in *Proc. IEEE Conf. Comput. Commun. Workshops*, 2015, pp. 185–190.
- [10] S. Jin, H. Wang, Z. Wang, F. Xiao, J. Hu, Y. He, W. Zhang, Z. Ba, W. Fang, S. Yuan *et al.*, "FaceObfuscator: Defending deep learning-based privacy attacks with gradient descent-resistant features in face recognition," in *Proc. USENIX Secur. Symp.*, 2024, pp. 6849–6866.
- [11] W. Zhu, Y. Sun, J. Liu, Y. Cheng, X. Ji, and W. Xu, "Campro: Camera-based anti-facial recognition," in *Proc. Netw. Distrib. Syst. Secur. Symp.*, 2024.
- [12] H. Zhang, X. Dong, Y. Lai, Y. Zhou, X. Zhang, X. Lv, Z. Jin, and X. Li, "Validating privacy-preserving face recognition under a minimum assumption," in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit.*, 2024, pp. 12 205–12 214.
- [13] R. Hasan, D. Crandall, M. Fritz, and A. Kapadia, "Automatically detecting bystanders in photos to reduce privacy risks," in *Proc. IEEE Symp. Secur. Privacy*, 2020, pp. 318–335.
- [14] D. Deb, J. Zhang, and A. K. Jain, "Advfaces: Adversarial face synthesis," in *Proc. IEEE Int. Joint Conf. Biometrics*, 2020, pp. 1–10.
- [15] X. Yang, Y. Dong, T. Pang, H. Su, J. Zhu, Y. Chen, and H. Xue, "Towards face encryption by generating adversarial identity masks," in *Proc. IEEE Int. Conf. Comput. Vis.*, 2021, pp. 3897–3907.
- [16] V. Cherepanova, M. Goldblum, H. Foley, S. Duan, J. P. Dickerson, G. Taylor, and T. Goldstein, "LowKey: Leveraging adversarial attacks to protect social media users from facial recognition," in *Proc. Int. Conf. Learn. Represent.*, 2021.
- [17] S. Shan, E. Wenger, J. Zhang, H. Li, H. Zheng, and B. Y. Zhao, "Fawkes: Protecting privacy against unauthorized deep learning models," in *Proc. USENIX Secur. Symp.*, 2020, pp. 1589–1604.
- [18] S. Hu, X. Liu, Y. Zhang, M. Li, L. Y. Zhang, H. Jin, and L. Wu, "Protecting facial privacy: Generating adversarial identity masks via style-robust makeup transfer," in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit.*, 2022, pp. 15 014–15 023.
- [19] S. Han, C. Lin, C. Shen, Q. Wang, and X. Guan, "Interpreting adversarial examples in deep learning: A review," *ACM Comput. Surv.*, vol. 55, no. 14s, pp. 1–38, 2023.

{15}------------------------------------------------

- [20] H. Jiang and D. Zeng, "Explainable face recognition based on accurate facial compositions," in *Proc. IEEE Int. Conf. Comput. Vis.*, 2021, pp. 1503–1512.
- [21] R. A. Johnston and A. J. Edmonds, "Familiar and unfamiliar face recognition: A review," *Mem.*, vol. 17, no. 5, pp. 577–596, 2009.
- [22] D. Li, W. Wang, K. Zhao, J. Dong, and T. Tan, "RiDDLE: Reversible and diversified de-identification with latent encryptor," in *Proc. IEEE Conf. Comput. Vis. Pattern Recognit.*, 2023.
- [23] Z. Cai, Z. Gao, B. Planche, M. Zheng, T. Chen, M. S. Asif, and Z. Wu, "Disguise without disruption: Utility-preserving face de-identification," in *Proc. AAAI Conf. Artif. Intell.*, 2024, pp. 918–926.
- [24] P. Sinha, B. Balas, Y. Ostrovsky, and R. Russell, "Face recognition by humans: Nineteen results all computer vision researchers should know about," *Proc. IEEE*, vol. 94, no. 11, pp. 1948–1962, 2006.
- [25] I. Evtimov, P. Sturmels, and T. Kohno, "Foggysight: A scheme for facial lookup privacy," in *Proc. Priv. Enhanc. Technol.*, 2021.
- [26] V. Chandrasekaran, C. Gao, B. Tang, K. Fawaz, S. Jha, and S. Banerjee, "Face-off: Adversarial face obfuscation," in *Proc. Priv. Enhanc. Technol.*, 2021.
- [27] Z. Li, N. Yu, A. Salem, M. Backes, M. Fritz, and Y. Zhang, "UnGAN-able: Defending against gan-based face manipulation," in *Proc. USENIX Secur. Symp.*, 2023, pp. 7213–7230.
- [28] Y. Zhong and W. Deng, "OPOM: Customized invisible cloak towards face privacy protection," *IEEE Trans. Pattern Anal. Mach. Intell.*, vol. 45, no. 3, pp. 3590–3603, 2023.
- [29] K.-H. Chow, S. Hu, T. Huang, and L. Liu, "Personalized privacy protection mask against unauthorized facial recognition," in *Proc. Eur. Conf. Comput. Vis.*, 2024, pp. 434–450.
- [30] S. Jia, B. Yin, T. Yao, S. Ding, C. Shen, X. Yang, and C. Ma, "Adv-attribute: Inconspicuous and transferable adversarial attack on face recognition," in *Proc. Neural Inf. Process. Syst.*, 2022, pp. 34 136–34 147.
- [31] Y. Sun, L. Yu, H. Xie, J. Li, and Y. Zhang, "Diffam: Diffusion-based adversarial makeup transfer for facial privacy protection," in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit.*, 2024, pp. 24 584–24 594.
- [32] Y. Lyu, Y. Jiang, Z. He, B. Peng, Y. Liu, and J. Dong, "3d-aware adversarial makeup generation for facial privacy protection," *IEEE Trans. Pattern Anal. Mach. Intell.*, 2023.
- [33] S. An, Y. Yao, Q. Xu, S. Ma, G. Tao, S. Cheng, K. Zhang, Y. Liu, G. Shen, I. Kelk, and X. Zhang, "ImU: Physical impersonating attack for face recognition system with natural style changes," in *Proc. IEEE Symp. Secur. Privacy*, 2023.
- [34] M. Li, J. Wang, H. Zhang, Z. Zhou, S. Hu, and X. Pei, "Transferable adversarial facial images for privacy protection," in *Proc. ACM Int. Conf. Multimedia*, 2024, p. 10649–10658.
- [35] M.-H. Le and N. Carlsson, "Diffprivate: Facial privacy protection with diffusion models," in *Proc. Priv. Enhanc. Technol.*, 2025, pp. 54–70.
- [36] W. Fan, M. Zhang, H. Li, W. Jiang, H. Chen, X. Yue, M. Backes, and X. Zhang, "Divtrackee versus dyntracker: Promoting diversity in anti-facial recognition against dynamic FR strategy," in *Proc. ACM Conf. Comput. Commun. Secur.*, 2025.
- [37] M. Wang, G. Hua, S. Li, and G. Feng, "A key-driven framework for identity-preserving face anonymization," in *Proc. Netw. Distrib. Syst. Secur. Symp.*, 2025.
- [38] J.-W. Chen, L.-J. Chen, C.-M. Yu, and C.-S. Lu, "Perceptual indistinguishability-net (PI-Net): Facial image obfuscation with manipulable semantics," in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit.*, 2021, pp. 6478–6487.
- [39] J. Li, L. Han, R. Chen, H. Zhang, B. Han, L. Wang, and X. Cao, "Identity-preserving face anonymization via adaptively facial attributes obfuscation," in *Proc. ACM Int. Conf. Multimedia*, 2021, pp. 3891–3899.
- [40] H. Hukkelås, R. Mester, and F. Lindseth, "Deepprivacy: A generative adversarial network for face anonymization," in *Proc. Intl. Symp. Vis. Comput.* Springer, 2019, pp. 565–578.
- [41] J. Lopez, C. Hinojosa, H. Arguello, and B. Ghanem, "Privacy-preserving optics for enhancing protection in face de-identification," in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit.*, 2024, pp. 12 120–12 129.
- [42] T. Wang, Y. Zhang, X. Xiao, L. Yuan, Z. Xia, and J. Weng, "Make privacy renewable! generating privacy-preserving faces supporting cancelable biometric recognition," in *Proc. ACM Int. Conf. Multimedia*, 2024, pp. 10 268–10 276.
- [43] Y. Wen, B. Liu, J. Cao, R. Xie, and L. Song, "Divide and conquer: a two-step method for high quality face de-identification with model explainability," in *Proc. IEEE Int. Conf. Comput. Vis.*, 2023, pp. 5148–5157.
- [44] M. Maximov, I. Elezi, and L. Leal-Taixé, "CIAGAN: Conditional identity anonymization generative adversarial networks," in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit.*, 2020, pp. 5447–5456.
- [45] Z. Zhang, T. Wei, W. Zhou, H. Zhao, W. Zhang, and N. Yu, "FaceRSA: RSA-Aware facial identity cryptography framework," in *Proc. AAAI Conf. Artif. Intell.*, 2024, pp. 7423–7431.
- [46] T. Wang, W. Wen, X. Xiao, Z. Hua, Y. Zhang, and Y. Fang, "Beyond privacy: Generating privacy-preserving faces supporting robust image authentication," *IEEE Trans. Inf. Forensics Security*, vol. 20, pp. 2564–2576, 2025.
- [47] K. Yang, J. H. Yau, L. Fei-Fei, J. Deng, and O. Russakovsky, "A study of face obfuscation in imagenet," in *Proc. Int. Conf. Mach. Learn.*, 2022, pp. 25 313–25 330.
- [48] S. B. H. Pias, I. Ahmad, T. Akter, A. Kapadia, and A. J. Lee, "Decaying photos for enhanced privacy: User perceptions towards temporal redactions and trusted platforms," *Proc. ACM Hum.-Comput. Interact.*, vol. 6, no. CSCW2, pp. 1–30, 2022.
- [49] T. Xie, H. Han, S. Shan, and X. Chen, "Natural adversarial mask for face identity protection in physical world," *IEEE Trans. Pattern Anal. Mach. Intell.*, vol. 47, no. 3, pp. 2089–2106, 2025.
- [50] F. Zhou, Q. Zhou, B. Yin, H. Zheng, X. Lu, L. Ma, and H. Ling, "Rethinking impersonation and dodging attacks on face recognition systems," in *Proc. ACM Int. Conf. Multimedia*, 2024, pp. 2487–2496.
- [51] Z. Li, B. Yin, T. Yao, J. Guo, S. Ding, S. Chen, and C. Liu, "Sibling-Attack: Rethinking transferable adversarial attacks against face recognition," in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit.*, 2023, pp. 24 626–24 637.
- [52] R. Shin and D. Song, "Jpeg-resistant adversarial images," in *Proc. Workshop Mach. Learn. Comput. Secur.*, vol. 1, 2017, p. 8.
- [53] F. Schroff, D. Kalenichenko, and J. Philbin, "FaceNet: A unified embedding for face recognition and clustering," in *Proc. IEEE Conf. Comput. Vis. Pattern Recognit.*, 2015, pp. 815–823.
- [54] K. He, X. Zhang, S. Ren, and J. Sun, "Deep residual learning for image recognition," in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit.*, 2016, pp. 770–778.
- [55] F. Shamshad, M. Naseer, and K. Nandakumar, "CLIP2Protect: Protecting facial privacy using text-guided makeup via adversarial latent search," in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit.*, 2023, pp. 20 595–20 605.
- [56] R. Meng, C. Yi, Y. Yu, S. Yang, B. Shen, and A. C. Kot, "Semantic deep hiding for robust unlearnable examples," *IEEE Trans. Inf. Forensics Security*, vol. 19, pp. 6545–6558, 2024.
- [57] Z. Kuang, X. Yang, Y. Shen, C. Hu, and J. Yu, "Facial identity anonymization via intrinsic and extrinsic attention distraction," in *Proc. IEEE Conf. Comput. Vis. Pattern Recognit.*, 2024, pp. 12 406–12 415.
- [58] S. Barattin, C. Tzelepis, I. Patras, and N. Sebe, "Attribute-preserving face dataset anonymization via latent code optimization," in *Proc. IEEE Conf. Comput. Vis. Pattern Recognit.*, 2023, pp. 8001–8010.
- [59] A. F. Chapman, H. Hawkins-Elder, and T. Susilo, "How robust is familiar face recognition? a repeat detection study of more than 1000 faces," *R. Soc. Open Sci.*, vol. 5, no. 5, p. 170634, 2018.
- [60] R. S. Kramer, A. W. Young, and A. M. Burton, "Understanding face familiarity," *Cogn.*, vol. 172, pp. 46–58, 2018.
- [61] Y. Hu, L. Manikonda, and S. Kambhampati, "What we instagram: A first analysis of instagram photo content and user types," in *Proc. Int. AAAI Conf. Web Social Media*, vol. 8, no. 1, 2014, pp. 595–598.
- [62] A. Khosla, A. Das Sarma, and R. Hamid, "What makes an image popular?" in *Proc. Int. Conf. World Wide Web*, 2014, pp. 867–876.
- [63] N. Zhang, M. Paluri, Y. Taigman, R. Fergus, and L. Bourdev, "Beyond frontal faces: Improving person recognition using multiple cues," in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit.*, 2015, pp. 4804–4813.
- [64] A. M. Burton, S. Wilson, M. Cowan, and V. Bruce, "Face recognition in poor-quality video: Evidence from security surveillance," *Psychological Science*, vol. 10, no. 3, pp. 243–248, 1999.
- [65] M. Ramon and M. I. Gobbini, "Familiarity matters: A review on prioritized processing of personally familiar faces," *Vis. Cogn.*, vol. 26, no. 3, pp. 179–195, 2018.
- [66] L. Backstrom, C. Dwork, and J. Kleinberg, "Wherefore art thou r3579x? anonymized social networks, hidden patterns, and structural steganography," in *Proc. Int. Conf. World Wide Web*, 2007, pp. 181–190.
- [67] S. J. Oh, R. Benenson, M. Fritz, and B. Schiele, "Faceless person recognition: Privacy implications in social media," in *Proc. Europ. Conf. Comp. Visi.* Springer, 2016, pp. 19–35.
- [68] R. Chen, X. Chen, B. Ni, and Y. Ge, "Simswap: An efficient framework for high fidelity face swapping," in *Proc. ACM Int. Conf. Multimedia*, 2020, pp. 2003–2011.

{16}------------------------------------------------

- [69] I. Gulrajani, F. Ahmed, M. Arjovsky, V. Dumoulin, and A. Courville, "Improved training of wasserstein gans," in *Proc. Neural Inf. Process. Syst.*, 2017, p. 5769–5779.
- [70] R. Zhang, P. Isola, A. A. Efros, E. Shechtman, and O. Wang, "The unreasonable effectiveness of deep features as a perceptual metric," in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit.*, 2018, pp. 586–595.
- [71] N. Abudarham and G. Yovel, "Reverse engineering the face space: Discovering the critical features for face identification," *J. Vision*, vol. 16, no. 3, pp. 40–40, 2016.

![Portrait of Lin Yuan, a man with short dark hair, wearing a dark suit and blue tie.](6cc4a2d5ea0462e4825d57bd689bd2b3_img.jpg)

Portrait of Lin Yuan, a man with short dark hair, wearing a dark suit and blue tie.

**Lin Yuan** received the B.Eng. degree in electronic science and technology from the University of Electronic Science and Technology of China (UESTC) in 2011 and the Ph.D. degree in electrical engineering from École Polytechnique Fédérale de Lausanne (EPFL), Switzerland, in 2017. He is currently an Associate Professor with the School of Cyber Security and Information Law, Chongqing University of Posts and Telecommunications. His research interests include image and video analysis, multimedia privacy protection, and media forensics.

![Portrait of Tao Wang, a man with short dark hair, wearing a white hoodie.](53cb558495f8cc93c4202314ef492966_img.jpg)

Portrait of Tao Wang, a man with short dark hair, wearing a white hoodie.

**Tao Wang** received the B.E. degree from the School of Computer and Information Technology, Anhui Normal University, Wuhu, China, in 2021 and the M.S. degree from the College of Computer Science and Technology, Nanjing University of Aeronautics and Astronautics, Nanjing, China, in 2024, where he is currently pursuing the Ph.D. degree. He has published several papers in top venues, e.g., IEEE TIFS, TDSC, ACM MM, CSUR. His current research interests include visual privacy, AIGC, adversarial perturbation, information theoretical privacy.

![Portrait of Wenying Wen, a woman with long dark hair, wearing a black turtleneck.](f7bc9b0327ed4589a3faf9a7b3c92712_img.jpg)

Portrait of Wenying Wen, a woman with long dark hair, wearing a black turtleneck.

**Wenying Wen** received the M.S. degree in computational mathematics from the Inner Mongolia University of Technology, Hohhot, China, in 2010, and the Ph.D. degree in computational mathematics from Chongqing University, Chongqing, China, in 2013. She is currently a Professor with the School of Computing and Artificial Intelligence, Jiangxi University of Finance and Economics. Her research interests include image processing, and multimedia security, compressive sensing security, and blockchain.

![Portrait of Yushu Zhang, a man with short dark hair and glasses, wearing a dark suit and tie, with a blue digital background.](bcc4b9d57d1d23e256f09d0a0a81be73_img.jpg)

Portrait of Yushu Zhang, a man with short dark hair and glasses, wearing a dark suit and tie, with a blue digital background.

**Yushu Zhang** received the Ph.D. degree from the College of Computer Science, Chongqing University, Chongqing, China, in December 2014. He held various research positions with the City University of Hong Kong, Hong Kong; Southwest University, Chongqing; the University of Macau, Macau, China; and Deakin University, Geelong, VIC, Australia. He is currently a Professor in Jiangxi University of Finance and Economics. He is an Associate Editor of IEEE Transactions on Dependable and Secure Computing and IEEE Transactions on Network and

![Portrait of Yuming Fang, a man with short dark hair, wearing a dark suit and tie.](bc453b6a6e6a96bdafd56edb6ea9d14b_img.jpg)

Portrait of Yuming Fang, a man with short dark hair, wearing a dark suit and tie.

**Yuming Fang** received the B.E. degree from Sichuan University, Chengdu, China, the M.S. degree from the Beijing University of Technology, Beijing, China, and the Ph.D. degree from Nanyang Technological University, Singapore. He is currently a Professor with the School of Information Management, Jiangxi University of Finance and Economics, Nanchang, China. His research interests include visual attention modeling, visual quality assessment, computer vision, and 3-D image/video processing.

Service Management, and Signal Processing. His research interests include multi media security, artificial intelligence security, and blockchain.

![Portrait of Xiangli Xiao, a man with short dark hair and glasses, wearing a black t-shirt.](3057ab3281cedf0b145d9b5f8e147e99_img.jpg)

Portrait of Xiangli Xiao, a man with short dark hair and glasses, wearing a black t-shirt.

**Xiangli Xiao** received the Ph.D. degree in cyberspace security from the College of Computer Science and Technology, Nanjing University of Aeronautics and Astronautics, Nanjing, China, in Oct. 2024. He is currently a Lecturer with the School of Computing and Artificial Intelligence, Jiangxi University of Finance and Economics, Nanchang, China. His current research interests include multimedia security, digital watermarking, and cloud computing security.

![Portrait of Kun Xu, a man with short dark hair, wearing a black t-shirt.](3d6d203581d8e268e252ca7f8476f829_img.jpg)

Portrait of Kun Xu, a man with short dark hair, wearing a black t-shirt.

**Kun Xu** received the B.E. and M.E. degrees from Anhui University of Science and Technology, Huainan, China, in 2020 and 2023, respectively. He is currently working toward the Ph.D. degree with the Nanjing University of Aeronautics and Astronautics, Nanjing, China. His research interests include generative model security.