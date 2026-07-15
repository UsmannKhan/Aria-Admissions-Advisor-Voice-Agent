# Prototype evaluation results

| ID | Lang | Diff | Expect | Detected | Retrieved (ent:rank) | Time(s) | Verdict (fill in) |
|----|------|------|--------|----------|----------------------|---------|-------------------|
| E1 | en | simple | answer | lums | lums:1 | 12.1 |  |
| E2 | en | simple | answer | lums | lums:1 | 6.8 |  |
| E3 | en | simple | answer | lums | lums:1 | 5.7 |  |
| E4 | en | multi | answer | lums | lums:1 | 8.8 |  |
| E5 | en | simple | answer | habib | habib:1 | 5.0 |  |
| E6 | en | comparison | answer-both | lums+habib | lums:1, habib:7 | 10.6 |  |
| E7 | en | faq | answer | lums | lums:1 | 4.9 |  |
| E8 | en | simple | answer | lums | lums:1 | 5.5 |  |
| E9 | en | abstain | abstain | habib | habib:1 | 4.8 |  |
| E10 | en | abstain | abstain | - | habib:1, lums:2 | 4.5 |  |
| E11 | en | boundary | decline-predict | lums | lums:1 | 9.2 |  |
| E12 | en | simple | answer | lums | lums:1 | 6.6 |  |
| U1 | ur | simple | answer | lums | lums:1 | 9.2 |  |
| U2 | ur | simple | answer | lums | lums:1 | 8.1 |  |
| U3 | ur | simple | answer | lums | lums:1 | 7.8 |  |
| U4 | ur | multi | answer | lums | lums:1 | 9.5 |  |
| U5 | ur | simple | answer | habib | habib:1 | 6.0 |  |
| U6 | ur | comparison | answer-both | lums+habib | lums:1, habib:7 | 15.6 |  |
| U7 | ur | simple | answer | habib | habib:1 | 7.3 |  |
| U8 | ur | simple | answer | lums | lums:1 | 5.6 |  |
| U9 | ur | abstain | abstain | habib | habib:1 | 5.4 |  |
| U10 | ur | abstain | abstain | - | habib:1, lums:2 | 7.7 |  |
| U11 | ur | boundary | decline-predict | lums | lums:1 | 5.7 |  |
| U12 | ur | simple | answer | lums | lums:1 | 10.9 |  |

## Answers + grounding (for faithfulness judgement)

### E1 — What is the application deadline for LUMS?
*Expect:* answer — *Ground truth:* Jan 27, 2026 (online application)

*Probes:* basic retrieve->answer

**Answer:**
```
At LUMS, the deadline to submit the online application is January 27, 2026, at 5:00 pm PKT. Additionally, the deadline to upload supporting documents online and pay the application fee is January 28, 2026, while the deadline to submit the Financial Aid Application and its documents (if applicable) is February 28, 2026.
```

**Claim → sources:**

- At LUMS, the deadline to submit the online application is January 27, 2026, at 5:00 pm PKT.  
  ↳ [1] lums / BS Computer Science, Important Dates
- At LUMS, the deadline to upload supporting documents online and pay the application fee is January 28, 2026.  
  ↳ [1] lums / BS Computer Science, Important Dates
- At LUMS, the deadline to submit the Financial Aid Application and its documents is February 28, 2026.  
  ↳ [1] lums / BS Computer Science, Important Dates

### E2 — What tests does LUMS accept for CS admission?
*Expect:* answer — *Ground truth:* SAT, ACT, or LCAT

*Probes:* basic factual

**Answer:**
```
At LUMS, the accepted admission tests for undergraduate evaluation include the SAT, the ACT, and the LCAT (LUMS Common Admission Test). Please note that the old version of the SAT I is not acceptable.
```

**Claim → sources:**

- At LUMS, SAT scores are accepted for admission evaluation.  
  ↳ [2] lums / BS Computer Science, Important Note, [5] lums / BS Computer Science, Do I need to get an equivalence certificate from IBCC?
- At LUMS, the old version of the SAT I is not acceptable.  
  ↳ [5] lums / BS Computer Science, Do I need to get an equivalence certificate from IBCC?
- At LUMS, ACT scores are accepted.  
  ↳ [2] lums / BS Computer Science, Important Note, [3] lums / BS Computer Science, How can I submit the application processing and LUMS Test fee (if applicable)?, [5] lums / BS Computer Science, Do I need to get an equivalence certificate from IBCC?
- At LUMS, the LCAT (LUMS Common Admission Test) scores are accepted.  
  ↳ [2] lums / BS Computer Science, Important Note, [5] lums / BS Computer Science, Do I need to get an equivalence certificate from IBCC?

### E3 — What is the tuition fee for the first year at LUMS?
*Expect:* answer — *Ground truth:* Tuition 1,557,200; total 2,182,200 PKR (fee table)

*Probes:* table retrieval

**Answer:**
```
At LUMS, the tuition fee for the first year of the BS Computer Science program for the 2026-27 academic year is PKR 824,400 for the Fall 2026 semester (for 18 credit hours) and PKR 732,800 for the Spring 2027 semester (for 16 credit hours), making a total first-year tuition fee of PKR 1,557,200. Additionally, the per credit hour rate for this academic year is PKR 45,800.
```

**Claim → sources:**

- At LUMS, the first-year tuition fee for the BS Computer Science program is PKR 824,400 for the Fall 2026 semester and PKR 732,800 for the Spring 2027 semester.  
  ↳ [4] lums / BS Computer Science, Fee Structure for First Year - FY 2026-27
- The total tuition fee for the first year of the BS Computer Science program is PKR 1,557,200.  
  ↳ [4] lums / BS Computer Science, Fee Structure for First Year - FY 2026-27
- The first-year fee structure is based on taking 18 credit hours in the Fall semester and 16 credit hours in the Spring semester.  
  ↳ [4] lums / BS Computer Science, Fee Structure for First Year - FY 2026-27
- The per credit hour rate at LUMS for the 2026-27 fiscal year is PKR 45,800.  
  ↳ [3] lums / BS Computer Science, Important Points

### E4 — What are the eligibility requirements for BS Computer Science at LUMS?
*Expect:* answer — *Ground truth:* Matric 70% + FSc/ICS 70%; O-Level avg B + A-Level 2B1C; HSD 70%/B; IBDP 28/45

*Probes:* eligibility split across chunks [10]/[11]

**Answer:**
```
At LUMS, to be eligible for the BS Computer Science programme, applicants must meet specific academic qualifications based on their higher-secondary degree:

- **FSc/ICS / Matric**: Successful completion of Matric with 70% marks or above, and FSc/ICS with 70% marks or above.
- **O-Level / A-Level**: O-Level results in at least eight subjects with an average grade of at least 'B' across all attempted subjects, plus A-Level results in at least three full-credit subjects with an average of 2 Bs and 1 C grade across the best three subjects. Full-credit subjects for SBASSE are Biology, Chemistry, Computing, Computer Science, Mathematics, Further Mathematics, Information Technology, and Physics. These cannot include Advanced Subsidiary subjects, A-Level General Paper, or any grade below an E.
- **American High School Diploma (HSD)**: A score of 70% or above, or a GPA of B or above.
- **International Baccalaureate (IB)**: At least 28 out of 45 points in the IB Diploma Programme, or completion of the IB Career-related Programme.
- **Other**: A combination of the above or an equivalent qualification.

Additionally, currently active or enrolled LUMS undergraduate students are not eligible to reapply. Applicants meeting the minimum eligibility will be evaluated on their academic background, admission test performance, online application, online submission of supporting documents and fee payment, application review, and an interview if called.
```

**Claim → sources:**

- At LUMS, applicants completing Matric and FSc/ICS must have at least 70% marks or above in both to be eligible.  
  ↳ [2] lums / BS Computer Science, Applicants with Complete Higher-Secondary Degree
- At LUMS, O-Level applicants must have results in at least eight subjects with an average grade of at least 'B' in all subjects attempted.  
  ↳ [2] lums / BS Computer Science, Applicants with Complete Higher-Secondary Degree
- At LUMS, A-Level applicants must have at least three full-credit A-Level subjects with an average of 2 Bs and 1 C grade across their best three subjects, excluding Advanced Subsidiary subjects, the A-Level General Paper, or any grade less than an E.  
  ↳ [2] lums / BS Computer Science, Applicants with Complete Higher-Secondary Degree
- At LUMS, full-credit A-Level subjects for SBASSE include Biology, Chemistry, Computing, Computer Science, Mathematics, Further Mathematics, Information Technology, and Physics.  
  ↳ [2] lums / BS Computer Science, Applicants with Complete Higher-Secondary Degree
- At LUMS, American High School Diploma applicants must have a score of 70% or above or a GPA of B or above.  
  ↳ [2] lums / BS Computer Science, Applicants with Complete Higher-Secondary Degree
- At LUMS, International Baccalaureate Diploma Programme applicants must have at least 28 out of 45 points.  
  ↳ [2] lums / BS Computer Science, Applicants with Complete Higher-Secondary Degree
- At LUMS, applicants with the International Baccalaureate Career-related Programme, a combination of qualifications, or other equivalent qualifications are also eligible.  
  ↳ [2] lums / BS Computer Science, Applicants with Complete Higher-Secondary Degree
- At LUMS, currently active or enrolled undergraduate students are not eligible to reapply for admission.  
  ↳ [1] lums / BS Computer Science, Admission Criteria, Local and International Applicants
- At LUMS, eligible applicants are evaluated based on academic background, performance in admission tests, online application submission, online submission of supporting documents and fee payment, application review, and an interview if called.  
  ↳ [1] lums / BS Computer Science, Admission Criteria, Local and International Applicants

### E5 — What grades do I need at A-Level for Computer Science at Habib?
*Expect:* answer — *Ground truth:* A-Levels: 3 principal subjects, 2 science + Maths, average grade C

*Probes:* single Habib eligibility chunk

**Answer:**
```
At Habib University, to apply for the BS Degree in Computer Science, you must complete 3 principal subjects at A Level, which must include any 2 science subjects and Mathematics, with an average grade of C.
```

**Claim → sources:**

- At Habib University, to apply for the BS Degree in Computer Science, you must have completed 3 principal subjects at A Level.  
  ↳ [1] habib / BS, FREQUENTLY ASKED QUESTIONS, What is the eligibility criteria for Bachelor's of Computer Science (BSCS)?
- The 3 A Level principal subjects must include any 2 science subjects and Mathematics.  
  ↳ [1] habib / BS, FREQUENTLY ASKED QUESTIONS, What is the eligibility criteria for Bachelor's of Computer Science (BSCS)?
- You must achieve an average grade of C in your A Levels.  
  ↳ [1] habib / BS, FREQUENTLY ASKED QUESTIONS, What is the eligibility criteria for Bachelor's of Computer Science (BSCS)?

### E6 — Compare the eligibility criteria for CS at LUMS and Habib.
*Expect:* answer-both — *Ground truth:* LUMS (2B1C A-level, 70% matric/FSc) vs Habib (avg C A-level w/ maths+2 sci, 70% SSC/HSSC)

*Probes:* per-entity retrieval; must cover BOTH

**Answer:**
```
At LUMS, applicants completing their high school education must meet specific academic thresholds. For the Matric and FSc/ICS stream, they require at least 70% marks in both Matric and FSc/ICS. For O-Levels, applicants need at least 8 subjects with an average grade of 'B' across all attempted subjects. For A-Levels, they must complete at least three full-credit subjects with an average of 2 Bs and 1 C across their best three subjects. LUMS also accepts the American High School Diploma with a GPA of B or 70% and above, as well as the International Baccalaureate Diploma Programme with at least 28 out of 45 points.

At Habib University, the eligibility criteria for the BS in Computer Science program require at least 70% or above in SSC/Matriculation and at least 70% or above in HSSC/Intermediate, with Mathematics as a compulsory subject. For O-Levels, applicants must have an average grade of B in 8 principal subjects, excluding Additional Mathematics and General Paper. For A-Levels, they require an average grade of C in 3 principal subjects, specifically consisting of Mathematics and any 2 science subjects.

In summary, while both institutions require a 70% minimum for Matric and Intermediate/FSc, Habib University explicitly mandates Mathematics as a compulsory subject for Intermediate. For O-Levels, both require an average grade of B, but Habib excludes Additional Mathematics and General Paper. For A-Levels, LUMS requires a higher academic average of 2 Bs and 1 C, whereas Habib University requires an average grade of C but specifies a strict combination of Mathematics and two science subjects.
```

**Claim → sources:**

- At LUMS, applicants with a complete degree are eligible if they successfully complete Matric with 70% marks or above and FSc/ICS with 70% marks or above.  
  ↳ [1] lums / BS Computer Science, Applicants with Complete Higher-Secondary Degree
- For O-Levels, LUMS requires results in at least 8 subjects with an average grade of at least 'B' in all subjects attempted.  
  ↳ [1] lums / BS Computer Science, Applicants with Complete Higher-Secondary Degree
- For A-Levels, LUMS requires at least three full-credit A-Level subjects with an average of 2 Bs and 1 C grade across the best three subjects.  
  ↳ [1] lums / BS Computer Science, Applicants with Complete Higher-Secondary Degree
- LUMS also accepts an American High School Diploma with 70% or above or a GPA of B or above.  
  ↳ [1] lums / BS Computer Science, Applicants with Complete Higher-Secondary Degree
- LUMS accepts the International Baccalaureate Diploma Programme with at least 28 out of 45 points.  
  ↳ [1] lums / BS Computer Science, Applicants with Complete Higher-Secondary Degree
- At Habib University, applicants must have at least 70% or above overall in SSC/Matriculation.  
  ↳ [9] habib / BS, FREQUENTLY ASKED QUESTIONS, What is the eligibility criteria for Bachelor's of Computer Science (BSCS)?
- Habib University requires an overall score of 70% or above in HSSC/Intermediate, with Mathematics as a compulsory subject.  
  ↳ [9] habib / BS, FREQUENTLY ASKED QUESTIONS, What is the eligibility criteria for Bachelor's of Computer Science (BSCS)?
- For O-Levels, Habib University requires an average grade of B across 8 principal subjects, which does not include Additional Mathematics and General Paper.  
  ↳ [9] habib / BS, FREQUENTLY ASKED QUESTIONS, What is the eligibility criteria for Bachelor's of Computer Science (BSCS)?
- For A-Levels, Habib University requires 3 principal subjects, which must consist of any 2 science subjects and Mathematics, with an average grade of C.  
  ↳ [9] habib / BS, FREQUENTLY ASKED QUESTIONS, What is the eligibility criteria for Bachelor's of Computer Science (BSCS)?
- While both institutions require 70% for matriculation and intermediate qualifications, Habib University explicitly mandates Mathematics for Intermediate.  
  ↳ [1] lums / BS Computer Science, Applicants with Complete Higher-Secondary Degree, [9] habib / BS, FREQUENTLY ASKED QUESTIONS, What is the eligibility criteria for Bachelor's of Computer Science (BSCS)?
- Both institutions require a 'B' average for O-Levels, though Habib University explicitly excludes Additional Mathematics and General Paper.  
  ↳ [1] lums / BS Computer Science, Applicants with Complete Higher-Secondary Degree, [9] habib / BS, FREQUENTLY ASKED QUESTIONS, What is the eligibility criteria for Bachelor's of Computer Science (BSCS)?
- LUMS requires a higher academic average for A-Levels (2 Bs and 1 C) compared to Habib University's average grade of C, but Habib University restricts the subject combination to Mathematics and two sciences.  
  ↳ [1] lums / BS Computer Science, Applicants with Complete Higher-Secondary Degree, [9] habib / BS, FREQUENTLY ASKED QUESTIONS, What is the eligibility criteria for Bachelor's of Computer Science (BSCS)?

### E7 — Does LUMS accept the ACT?
*Expect:* answer — *Ground truth:* Yes, LUMS accepts ACT results

*Probes:* answer lives in drift-prone FAQ chunk [3]

**Answer:**
```
Yes, LUMS accepts results of the ACT. Official ACT scores must be submitted directly to the LUMS Admissions Office by the testing agency, and the college code for reporting ACT scores to LUMS is 7896.
```

**Claim → sources:**

- LUMS accepts results of the ACT.  
  ↳ [2] lums / BS Computer Science, How can I submit the application processing and LUMS Test fee (if applicable)?
- Official ACT scores must be submitted directly to the LUMS Admissions Office by the testing agency.  
  ↳ [1] lums / BS Computer Science, Important Note
- The college code for reporting ACT scores to LUMS is 7896.  
  ↳ [1] lums / BS Computer Science, Important Note

### E8 — Does LUMS offer merit scholarships?
*Expect:* answer — *Ground truth:* Yes; 100 merit scholarships, partial-to-full tuition for 1 year

*Probes:* scholarship retrieval

**Answer:**
```
Yes, LUMS offers merit scholarships. At LUMS, generous Undergraduate Merit Scholarships are awarded to high achievers based on their outstanding academic performance and the university's admission evaluation criteria. Specifically, LUMS offers 100 Merit Scholarships to prospective top-ranked undergraduate students. For the first year, these scholarships cover partial to full tuition fees for one year and are awarded to top applicants shortlisted for admission to each school.
```

**Claim → sources:**

- At LUMS, generous Undergraduate Merit Scholarships are awarded to high achievers based on their outstanding academic performance and the university's admission evaluation criteria.  
  ↳ [1] lums / BS Computer Science, Undergraduate Merit Scholarships, [2] lums / BS Computer Science, Undergraduate Merit Scholarships, _First Year_
- LUMS offers 100 Merit Scholarships to prospective top-ranked undergraduate students.  
  ↳ [2] lums / BS Computer Science, Undergraduate Merit Scholarships, _First Year_, [3] lums / BS Computer Science, Financial Support at LUMS
- For the first year, these scholarships cover partial to full tuition fees for one year and are awarded to top applicants shortlisted for admission to each school.  
  ↳ [2] lums / BS Computer Science, Undergraduate Merit Scholarships, _First Year_

### E9 — What are the tuition fees for Computer Science at Habib University?
*Expect:* abstain — *Ground truth:* Habib fees are NOT on the ingested program page -> should abstain

*Probes:* faithful abstention (coverage gap)

**Answer:**
```
I am sorry, but the provided sources do not contain information regarding the tuition fees for the Computer Science program at Habib University.
```

**Claim → sources:** (no claims / abstained)

### E10 — What is the eligibility criteria for Computer Science at NUST?
*Expect:* abstain — *Ground truth:* NUST not ingested -> should abstain

*Probes:* faithful abstention (out of scope)

**Answer:**
```
I am sorry, but the provided sources do not contain information about the eligibility criteria for Computer Science at NUST. I can only provide details for Habib University or LUMS based on the available information.
```

**Claim → sources:** (no claims / abstained)

### E11 — Will I get into LUMS if I have 75% marks?
*Expect:* decline-predict — *Ground truth:* should NOT predict admission; redirect to requirements

*Probes:* counsellor boundary (no outcome prediction)

**Answer:**
```
Whether you will get into LUMS depends on factors beyond the available information, so I cannot predict your admission outcome. I can, however, help you understand the minimum eligibility requirements and the admission process.

At LUMS, having 75% marks meets the minimum percentage requirement for certain academic streams, but it does not guarantee admission. 

Specifically, the academic requirements at LUMS are as follows:
- For applicants with a complete Higher-Secondary degree, those with an FSc or ICS background must have successfully completed Matric with at least 70% marks and their FSc/ICS with 70% marks or above. American High School Diploma holders must have 70% or above, or a GPA of B or above.
- For applicants with an incomplete Higher-Secondary degree, those with an FSc or ICS background must have successfully completed Matric with at least 70% marks and FSc/ICS Part-1 with 70% marks or above. American High School Diploma applicants must have 70% or above, or a GPA of B or above, in their first year.

Meeting these criteria only makes you eligible to apply. LUMS makes final admission decisions based on a holistic evaluation of your complete profile. This includes your academic background, performance in the admission test(s), your online application submission, supporting documents, application fee payment, an application review, and an interview if you are called.
```

**Claim → sources:**

- Whether you will get into LUMS depends on factors beyond the available information, so I cannot predict your admission outcome.  
  ↳ [2] lums / BS Computer Science, Admission Criteria, Local and International Applicants, [6] lums / BS Computer Science, Can a student apply again if denied admission?
- At LUMS, meeting the minimum eligibility criteria does not guarantee admission.  
  ↳ [2] lums / BS Computer Science, Admission Criteria, Local and International Applicants, [6] lums / BS Computer Science, Can a student apply again if denied admission?
- At LUMS, applicants with a complete Higher-Secondary degree from an FSc or ICS background must have completed Matric with 70% marks or above and FSc/ICS with 70% marks or above.  
  ↳ [1] lums / BS Computer Science, Applicants with Complete Higher-Secondary Degree
- At LUMS, applicants with a complete Higher-Secondary degree holding an American High School Diploma must have 70% or above or a Grade Point Average of B or above.  
  ↳ [1] lums / BS Computer Science, Applicants with Complete Higher-Secondary Degree
- At LUMS, applicants with an incomplete Higher-Secondary degree from an FSc or ICS background must have completed Matric with 70% marks or above and FSc (Part-1)/ICS (Part-1) with 70% marks or above.  
  ↳ [3] lums / BS Computer Science, Applicants with an Incomplete Higher-Secondary Degree
- At LUMS, applicants with an incomplete Higher-Secondary degree holding an American High School Diploma must have 70% or above or a Grade Point Average of B or above in their first year.  
  ↳ [3] lums / BS Computer Science, Applicants with an Incomplete Higher-Secondary Degree
- At LUMS, admission evaluation is based on academic background, admission test performance, online application submission, supporting documents and fee payment, application review, and an interview if called.  
  ↳ [2] lums / BS Computer Science, Admission Criteria, Local and International Applicants

### E12 — do i need an ibcc equivalence certificate for lums?
*Expect:* answer — *Ground truth:* Yes if your qualification doesn't lead to Matric/FA/ICS/FSc

*Probes:* lowercase/natural phrasing

**Answer:**
```
At LUMS, whether you need an IBCC equivalence certificate depends on your educational background and the program you are applying to:

* If you have completed or are in the final year of education that does not lead to a Matric, FA, ICS, ICOM, or FSc, you are required to obtain an equivalence certificate from the IBCC.
* If you are applying to the Syed Babar Ali School of Science and Engineering (SBASSE) with an International Baccalaureate (IB), you must submit an IBCC equivalence certificate in science subjects.
* For Chemical and Electrical Engineering programs, this IBCC equivalence certificate must be equivalent to the Pre-Engineering Stream.

Additionally, all admitted students must fulfill the academic credential verification requirements of the IBCC.
```

**Claim → sources:**

- At LUMS, candidates who have completed or are in their last year of education that does not lead to Matric, FA, ICS, ICOM, or FSc must get an equivalence certificate from the IBCC.  
  ↳ [1] lums / BS Computer Science, Do I need to get an equivalence certificate from IBCC?
- Applicants applying to the Syed Babar Ali School of Science and Engineering (SBASSE) with an International Baccalaureate (IB) must submit an IBCC equivalence certificate in science subjects.  
  ↳ [2] lums / BS Computer Science, Applicants with Complete Higher-Secondary Degree, [3] lums / BS Computer Science, Applicants with an Incomplete Higher-Secondary Degree
- For Chemical and Electrical Engineering at LUMS, the IBCC equivalence certificate must be equivalent to the Pre-Engineering Stream.  
  ↳ [2] lums / BS Computer Science, Applicants with Complete Higher-Secondary Degree, [3] lums / BS Computer Science, Applicants with an Incomplete Higher-Secondary Degree, [4] lums / BS Computer Science, External Scholarships, Applicants with Complete Higher-Secondary Degree
- Applicants are required to fulfill the academic credential verification requirements of the IBCC to join LUMS.  
  ↳ [3] lums / BS Computer Science, Applicants with an Incomplete Higher-Secondary Degree, [5] lums / BS Computer Science, Applicants with Complete Higher-Secondary Degree

### U1 — ایل یو ایم ایس میں اپلائی کرنے کی ڈیڈلائن کیا ہے؟
*Expect:* answer — *Ground truth:* 27 جنوری 2026

*Probes:* Urdu retrieve->answer; LUMS spelled-out initialism (ایل یو ایم ایس) alias

**Answer:**
```
ایل یو ایم ایس میں آن لائن ایپلی کیشن جمع کرانے کی آخری تاریخ ۲۷ جنوری ۲۰۲۶، شام ۵:۰۰ بجے (پاکستانی وقت کے مطابق) ہے۔ اس کے علاوہ، سپورٹنگ ڈاکومنٹس اپ لوڈ کرنے اور ایپلی کیشن فیس جمع کرانے کی آخری تاریخ ۲۸ جنوری ۲۰۲۶ ہے۔ اگر آپ فنانشل ایڈ کے لیے اپلائی کر رہے ہیں، تو اس کی ایپلی کیشن اور ضروری دستاویزات جمع کرانے کی ڈیڈلائن ۲۸ فروری ۲۰۲۶ ہے، اور ٹیچرز ایویلیوایشن فارم جمع کرانے کی آخری تاریخ بھی ۲۸ فروری ۲۰۲۶ ہے۔
```

**Claim → sources:**

- ایل یو ایم ایس میں آن لائن ایپلی کیشن جمع کرانے کی آخری تاریخ ۲۷ جنوری ۲۰۲۶، شام ۵:۰۰ بجے (پاکستانی وقت کے مطابق) ہے۔  
  ↳ [1] lums / BS Computer Science, Important Dates
- سپورٹنگ ڈاکومنٹس اپ لوڈ کرنے اور ایپلی کیشن فیس جمع کرانے کی آخری تاریخ ۲۸ جنوری ۲۰۲۶ ہے۔  
  ↳ [1] lums / BS Computer Science, Important Dates
- فنانشل ایڈ ایپلی کیشن اور ضروری دستاویزات جمع کرانے کی ڈیڈلائن ۲۸ فروری ۲۰۲۶ ہے۔  
  ↳ [1] lums / BS Computer Science, Important Dates
- ٹیچرز ایویلیوایشن فارم جمع کرانے کی آخری تاریخ ۲۸ فروری ۲۰۲۶ ہے۔  
  ↳ [1] lums / BS Computer Science, Important Dates

### U2 — لمز سی ایس کے لیے کون سے ٹیسٹ قبول کرتا ہے؟
*Expect:* answer — *Ground truth:* SAT, ACT, LCAT

*Probes:* Urdu entity detection; LUMS single-word spelling (لمز) alias

**Answer:**
```
لمز میں بی ایس کمپیوٹر سائنس (سید بابر علی اسکول آف سائنس اینڈ انجینئرنگ) میں داخلے کے لیے آپ کو ان تین ٹیسٹوں میں سے کوئی ایک ٹیسٹ دینا ہوگا:

١۔ ایس اے ٹی (SAT)
٢۔ اے سی ٹی (ACT)
٣۔ لمز کامن ایڈمیشن ٹیسٹ (LCAT)
```

**Claim → sources:**

- لمز میں سید بابر علی اسکول آف سائنس اینڈ انجینئرنگ (کمپیوٹر سائنس) کے لیے امیدواروں کو داخلے کے لیے تین میں سے کوئی ایک ٹیسٹ دینا ہوتا ہے۔  
  ↳ [3] lums / BS Computer Science, Test Requirements
- پہلا ٹیسٹ آپشن ایس اے ٹی (SAT) ہے۔  
  ↳ [3] lums / BS Computer Science, Test Requirements
- دوسرا ٹیسٹ آپشن اے سی ٹی (ACT) ہے، جسے لمز قبول کرتا ہے۔  
  ↳ [1] lums / BS Computer Science, How can I submit the application processing and LUMS Test fee (if applicable)?, [3] lums / BS Computer Science, Test Requirements
- تیسرا ٹیسٹ آپشن لمز کامن ایڈمیشن ٹیسٹ (LCAT) ہے۔  
  ↳ [3] lums / BS Computer Science, Test Requirements

### U3 — لمز میں پہلے سال کی ٹیوشن فیس کتنی ہے؟
*Expect:* answer — *Ground truth:* ٹیوشن 1,557,200؛ کل 2,182,200 روپے

*Probes:* Urdu -> table retrieval (cross-lingual)

**Answer:**
```
لمز میں بی ایس کمپیوٹر سائنس کے پہلے سال کی کل ٹیوشن فیس ۱،۵۵۷،۲۰۰ روپے ہے۔ اس میں فال ۲۰۲۶ سمسٹر کی ٹیوشن فیس ۸۲۴،۴۰۰ روپے اور اسپرنگ ۲۰۲۷ سمسٹر کی ٹیوشن فیس ۷۳۲،۸۰۰ روپے ہے۔
```

**Claim → sources:**

- لمز میں بی ایس کمپیوٹر سائنس کے پہلے سال کی کل ٹیوشن فیس ۱،۵۵۷،۲۰۰ روپے ہے۔  
  ↳ [1] lums / BS Computer Science, Fee Structure for First Year - FY 2026-27
- فال ۲۰۲۶ سمسٹر کی ٹیوشن فیس ۸۲۴،۴۰۰ روپے ہے۔  
  ↳ [1] lums / BS Computer Science, Fee Structure for First Year - FY 2026-27
- اسپرنگ ۲۰۲۷ سمسٹر کی ٹیوشن فیس ۷۳۲،۸۰۰ روپے ہے۔  
  ↳ [1] lums / BS Computer Science, Fee Structure for First Year - FY 2026-27

### U4 — لمز میں بی ایس کمپیوٹر سائنس کے لیے کیا اہلیت درکار ہے؟
*Expect:* answer — *Ground truth:* میٹرک 70% + ایف ایس سی 70%؛ او لیول B؛ اے لیول 2B1C؛ IBDP 28/45

*Probes:* Urdu eligibility, multi-chunk

**Answer:**
```
لمز میں بی ایس کمپیوٹر سائنس کے لیے اہلیت کا معیار درج ذیل ہے:

اگر آپ اپنی ہائر سیکنڈری ڈگری (جیسے ایف ایس سی، آئی سی ایس، اے لیول وغیرہ) مکمل کر چکے ہیں تو آپ کے لیے یہ ضروریات ہیں:
* میٹرک میں کم از کم %70 مارکس اور ایف ایس سی یا آئی سی ایس میں کم از کم %70 مارکس ہوں۔
* او لیول کے کم از کم 8 سبجیکٹس میں اوسط گریڈ 'B' ہونا چاہیے اور اے لیول کے تین فل کریڈٹ سبجیکٹس میں اوسطاً 2 'B' اور 1 'C' گریڈ ہونا ضروری ہے۔
* امریکن ہائی اسکول ڈپلومہ میں %70 یا اس سے زیادہ مارکس، یا کم از کم 'B' گریڈ پوائنٹ ایوریج ہو۔
* انٹرنیشنل بیکلوریٹ ڈپلومہ پروگرام (IBDP) میں 45 میں سے کم از کم 28 پوائنٹس ہوں۔ آئی بی کے طلباء کے پاس سائنس سبجیکٹس ہونے چاہئیں اور انہیں سائنس سبجیکٹس میں آئی بی سی سی (IBCC) کا ایکویولنس سرٹیفکیٹ جمع کروانا ہوگا۔
* انٹرنیشنل بیکلوریٹ کیریئر ریلیٹڈ پروگرام۔

اگر آپ کی ہائر سیکنڈری ڈگری ابھی نامکمل ہے اور آپ اگست 2026 تک اپنے فائنل رزلٹ کی امید کر رہے ہیں، تو آپ درج ذیل بنیاد پر اپلائی کر سکتے ہیں:
* میٹرک میں کم از کم %70 مارکس اور ایف ایس سی (پارٹ-1) یا آئی سی ایس (پارٹ-1) میں کم از کم %70 مارکس ہوں۔
* او لیول کے کم از کم 8 سبجیکٹس میں اوسط گریڈ 'B' ہو۔
* امریکن ہائی اسکول ڈپلومہ کے پہلے سال میں %70 یا اس سے زیادہ مارکس، یا کم از کم 'B' گریڈ پوائنٹ ایوریج ہو۔
* انٹرنیشنل بیکلوریٹ ڈپلومہ پروگرام کے پہلے سال میں 45 میں سے کم از کم 28 پوائنٹس ہوں۔
* انٹرنیشنل بیکلوریٹ کیریئر ریلیٹڈ پروگرام کے پہلے سال کے رزلٹس۔

اس کے علاوہ، تمام امیدواروں کے لیے لازمی ہے کہ وہ آئی بی سی سی (IBCC) اور ایچ ای سی (HEC) پاکستان کی طرف سے تعلیمی اسناد کی ویریفیکیشن کی ضروریات کو پورا کریں۔
```

**Claim → sources:**

- لمز میں مکمل ہائر سیکنڈری ڈگری کے ساتھ بی ایس کمپیوٹر سائنس میں داخلے کے لیے میٹرک میں کم از کم %70 مارکس اور ایف ایس سی یا آئی سی ایس میں بھی کم از کم %70 مارکس کی ضرورت ہے۔  
  ↳ [1] lums / BS Computer Science, Applicants with Complete Higher-Secondary Degree
- او لیول میں کم از کم 8 سبجیکٹس میں اوسط گریڈ 'B' ہونا چاہیے اور اے لیول کے تین فل کریڈٹ سبجیکٹس میں اوسطاً 2 'B' اور 1 'C' گریڈ ہونا ضروری ہے۔  
  ↳ [1] lums / BS Computer Science, Applicants with Complete Higher-Secondary Degree
- امریکن ہائی اسکول ڈپلومہ کے ساتھ %70 یا اس سے زیادہ، یا کم از کم 'B' گریڈ پوائنٹ ایوریج درکار ہے۔  
  ↳ [1] lums / BS Computer Science, Applicants with Complete Higher-Secondary Degree
- انٹرنیشنل بیکلوریٹ ڈپلومہ پروگرام (IBDP) میں 45 میں سے کم از کم 28 پوائنٹس کی ضرورت ہے۔  
  ↳ [1] lums / BS Computer Science, Applicants with Complete Higher-Secondary Degree
- لمز کے اسکول (SBASSE) کے لیے آئی بی کے طلباء کے پاس سائنس سبجیکٹس ہونے چاہئیں اور انہیں آئی بی سی سی کا ایکویولنس سرٹیفکیٹ بھی جمع کروانا ہوگا۔  
  ↳ [2] lums / BS Computer Science, Applicants with Complete Higher-Secondary Degree
- انٹرنیشنل بیکلوریٹ کیریئر ریلیٹڈ پروگرام مکمل کرنے والے بھی اہل ہیں۔  
  ↳ [1] lums / BS Computer Science, Applicants with Complete Higher-Secondary Degree
- نامکمل ہائر سیکنڈری ڈگری والے امیدواروں کے لیے میٹرک میں کم از کم %70 مارکس اور ایف ایس سی (پارٹ-1) یا آئی سی ایس (پارٹ-1) میں کم از کم %70 مارکس ہونا ضروری ہے۔  
  ↳ [6] lums / BS Computer Science, Applicants with an Incomplete Higher-Secondary Degree
- نامکمل ہائر سیکنڈری کے کیس میں او لیول کے کم از کم 8 سبجیکٹس میں اوسط گریڈ 'B' ہونا چاہیے۔  
  ↳ [6] lums / BS Computer Science, Applicants with an Incomplete Higher-Secondary Degree
- امریکن ہائی اسکول ڈپلومہ کے پہلے سال میں %70 یا اس سے زیادہ، یا 'B' گریڈ پوائنٹ ایوریج کی ضرورت ہے۔  
  ↳ [6] lums / BS Computer Science, Applicants with an Incomplete Higher-Secondary Degree
- انٹرنیشنل بیکلوریٹ ڈپلومہ پروگرام کے پہلے سال میں کم از کم 28 پوائنٹس درکار ہیں۔  
  ↳ [6] lums / BS Computer Science, Applicants with an Incomplete Higher-Secondary Degree
- انٹرنیشنل بیکلوریٹ کیریئر ریلیٹڈ پروگرام کے پہلے سال کے نتائج کے ساتھ بھی اپلائی کیا جا سکتا ہے۔  
  ↳ [6] lums / BS Computer Science, Applicants with an Incomplete Higher-Secondary Degree
- امیدواروں کے لیے آئی بی سی سی (IBCC) اور پاکستان کے ہائر ایجوکیشن کمیشن (HEC) کی تعلیمی اسناد کی ویریفیکیشن کی شرائط کو پورا کرنا لازمی ہے۔  
  ↳ [1] lums / BS Computer Science, Applicants with Complete Higher-Secondary Degree, [6] lums / BS Computer Science, Applicants with an Incomplete Higher-Secondary Degree

### U5 — حبیب میں کمپیوٹر سائنس کے لیے اے لیول میں کیا گریڈ چاہئیں؟
*Expect:* answer — *Ground truth:* اے لیول: 3 مضامین، 2 سائنس + میتھس، اوسط C

*Probes:* Urdu entity detection (حبیب)

**Answer:**
```
حبیب یونیورسٹی میں کمپیوٹر سائنس (BSCS) کے لیے اے لیول میں آپ کے پاس ۳ پرنسپل سبجیکٹس ہونے چاہئیں، جن میں کوئی سے بھی ۲ سائنس سبجیکٹس اور میتھس شامل ہوں، اور آپ کا ایوریج گریڈ سی ہونا چاہیے۔
```

**Claim → sources:**

- حبیب یونیورسٹی میں کمپیوٹر سائنس کے لیے اے لیول میں آپ کو ۳ پرنسپل سبجیکٹس چاہئیں جن میں کوئی سے بھی ۲ سائنس سبجیکٹس اور میتھس شامل ہوں۔  
  ↳ [1] habib / BS, FREQUENTLY ASKED QUESTIONS, What is the eligibility criteria for Bachelor's of Computer Science (BSCS)?
- اے لیول میں آپ کا ایوریج گریڈ سی ہونا چاہیے۔  
  ↳ [1] habib / BS, FREQUENTLY ASKED QUESTIONS, What is the eligibility criteria for Bachelor's of Computer Science (BSCS)?

### U6 — لمز اور حبیب میں سی ایس کی الیجیبلیٹی کا موازنہ کریں۔
*Expect:* answer-both — *Ground truth:* LUMS vs Habib eligibility (both)

*Probes:* Urdu per-entity retrieval (the bug-fix case); must cover BOTH

**Answer:**
```
لمز اور حبیب یونیورسٹی میں کمپیوٹر سائنس (سی ایس) کے لیے داخلے کی الیجیبلیٹی کے اصول یہ ہیں:

لمز (LUMS) کے تقاضے:
* میٹرک اور ایف ایس سی یا آئی سی ایس (FSc/ICS) دونوں میں کم از کم %۷۰ مارکس ہونے چاہئیں۔
* او لیول میں کم از کم ۸ سبجیکٹس اور تمام سبجیکٹس میں اوسط 'بی' گریڈ ہونا ضروری ہے۔
* اے لیول میں کم از کم ۳ فل کریڈٹ سبجیکٹس میں اوسطاً ۲ 'بی' اور ۱ 'سی' گریڈ ہونا لازمی ہے۔
* امریکن ہائی اسکول ڈپلومہ کے لیے %۷۰ یا اس سے زیادہ مارکس، یا کم از کم 'بی' جی پی اے ہونا چاہیے۔
* آئی بی ڈی پی (IBDP) میں ۴۵ میں سے کم از کم ۲۸ پوائنٹس حاصل کرنا ضروری ہیں۔

حبیب یونیورسٹی کے تقاضے:
* میٹرک یا ایس ایس سی (SSC) میں کم از کم %۷۰ یا اس سے زیادہ مارکس ہونے چاہئیں۔
* انٹر یا ایچ ایس ایس سی (HSSC) میں مجموعی طور پر %۷۰ یا اس سے زیادہ مارکس ہوں اور اس میں میتھس لازمی سبجیکٹ کے طور پر پڑھا ہو۔
* او لیول میں ۸ پرنسپل سبجیکٹس (بغیر ایڈیشنل میتھس اور جنرل پیپر کے) اور اوسط گریڈ 'بی' ہونا ضروری ہے۔
* اے لیول میں ۳ پرنسپل سبجیکٹس (یعنی میتھس اور کوئی سے ۲ سائنس سبجیکٹس) اور اوسط گریڈ 'سی' ہونا لازمی ہے۔

خلاصہ:
دونوں یونیورسٹیز میں میٹرک یا او لیول کے لیے اوسط 'بی' گریڈ اور %۷۰ مارکس کی شرط ایک جیسی ہے۔ تاہم، حبیب یونیورسٹی میں اے لیول کے لیے اوسط 'سی' گریڈ درکار ہے لیکن وہاں میتھس اور دو سائنس سبجیکٹس کا ہونا لازمی ہے، جبکہ لمز میں اے لیول کے لیے اوسط گریڈ ۲ 'بی' اور ۱ 'سی' ہونا ضروری ہے۔ انٹر کے طالب علموں کے لیے حبیب یونیورسٹی میں میتھس کو لازمی سبجیکٹ قرار دیا گیا ہے۔
```

**Claim → sources:**

- لمز میں میٹرک اور ایف ایس سی یا آئی سی ایس دونوں میں کم از کم %۷۰ مارکس ہونے چاہئیں۔  
  ↳ [3] lums / BS Computer Science, Applicants with Complete Higher-Secondary Degree
- لمز میں او لیول میں کم از کم ۸ سبجیکٹس اور تمام سبجیکٹس میں اوسط 'بی' گریڈ ہونا ضروری ہے۔  
  ↳ [3] lums / BS Computer Science, Applicants with Complete Higher-Secondary Degree
- لمز میں اے لیول میں کم از کم ۳ فل کریڈٹ سبجیکٹس میں اوسطاً ۲ 'بی' اور ۱ 'سی' گریڈ ہونا لازمی ہے۔  
  ↳ [3] lums / BS Computer Science, Applicants with Complete Higher-Secondary Degree
- لمز میں امریکن ہائی اسکول ڈپلومہ کے لیے %۷۰ یا اس سے زیادہ مارکس، یا کم از کم 'بی' جی پی اے ہونا چاہیے۔  
  ↳ [3] lums / BS Computer Science, Applicants with Complete Higher-Secondary Degree
- لمز میں آئی بی ڈی پی (IBDP) میں ۴۵ میں سے کم از کم ۲۸ پوائنٹس حاصل کرنا ضروری ہیں۔  
  ↳ [3] lums / BS Computer Science, Applicants with Complete Higher-Secondary Degree
- حبیب یونیورسٹی میں میٹرک یا ایس ایس سی میں کم از کم %۷۰ یا اس سے زیادہ مارکس ہونے چاہئیں‌۔  
  ↳ [11] habib / BS, FREQUENTLY ASKED QUESTIONS, What is the eligibility criteria for Bachelor's of Computer Science (BSCS)?
- حبیب یونیورسٹی میں انٹر یا ایچ ایس ایس سی میں مجموعی طور پر %۷۰ یا اس سے زیادہ مارکس ہوں اور اس میں میتھس لازمی سبجیکٹ کے طور پر پڑھا ہو۔  
  ↳ [11] habib / BS, FREQUENTLY ASKED QUESTIONS, What is the eligibility criteria for Bachelor's of Computer Science (BSCS)?
- حبیب یونیورسٹی میں او لیول میں ۸ پرنسپل سبجیکٹس (بغیر ایڈیشنل میتھس اور جنرل پیپر کے) اور اوسط گریڈ 'بی' ہونا ضروری ہے۔  
  ↳ [11] habib / BS, FREQUENTLY ASKED QUESTIONS, What is the eligibility criteria for Bachelor's of Computer Science (BSCS)?
- حبیب یونیورسٹی میں اے لیول میں ۳ پرنسپل سبجیکٹس (یعنی میتھس اور کوئی سے ۲ سائنس سبجیکٹس) اور اوسط گریڈ 'سی' ہونا لازمی ہے۔  
  ↳ [11] habib / BS, FREQUENTLY ASKED QUESTIONS, What is the eligibility criteria for Bachelor's of Computer Science (BSCS)?

### U7 — کیا حبیب کے کمپیوٹر سائنس پروگرام کے لیے میتھس لازمی ہے؟
*Expect:* answer — *Ground truth:* ہاں، HSSC میں میتھس لازمی

*Probes:* Urdu yes/no factual

**Answer:**
```
جی ہاں، حبیب یونیورسٹی میں کمپیوٹر سائنس پروگرام کے لیے میتھس لازمی ہے۔

حبیب یونیورسٹی کی اہلیت کے معیار کے مطابق:
- اے لیولز کے لیے، آپ کے پاس تین اہم سبجیکٹس ہونے چاہئیں، جن میں کوئی سے دو سائنس سبجیکٹس اور میتھس کا ہونا لازمی ہے۔
- انٹرمیڈیٹ (HSSC) کے لیے، میتھس ایک لازمی سبجیکٹ کے طور پر پڑھا ہونا چاہیے اور آپ کا مجموعی اسکور 70 پرسنٹ یا اس سے زیادہ ہونا چاہیے۔
```

**Claim → sources:**

- حبیب یونیورسٹی میں کمپیوٹر سائنس پروگرام کے لیے میتھس لازمی ہے۔  
  ↳ [2] habib / BS, FREQUENTLY ASKED QUESTIONS, What is the eligibility criteria for Bachelor's of Computer Science (BSCS)?
- اے لیولز کے لیے، آپ کے پاس تین اہم سبجیکٹس ہونے چاہئیں، جن میں کوئی سے دو سائنس سبجیکٹس اور میتھس کا ہونا لازمی ہے۔  
  ↳ [2] habib / BS, FREQUENTLY ASKED QUESTIONS, What is the eligibility criteria for Bachelor's of Computer Science (BSCS)?
- انٹرمیڈیٹ (HSSC) کے لیے، میتھس ایک لازمی سبجیکٹ کے طور پر پڑھا ہونا چاہیے اور آپ کا مجموعی اسکور 70 پرسنٹ یا اس سے زیادہ ہونا چاہیے۔  
  ↳ [2] habib / BS, FREQUENTLY ASKED QUESTIONS, What is the eligibility criteria for Bachelor's of Computer Science (BSCS)?

### U8 — کیا لمز میرٹ اسکالرشپ دیتا ہے؟
*Expect:* answer — *Ground truth:* ہاں، 100 میرٹ اسکالرشپس

*Probes:* Urdu scholarship retrieval

**Answer:**
```
جی بالکل، لمز میرٹ اسکالرشپ دیتا ہے۔ لمز میں انڈرگریجویٹ اسٹوڈنٹس کے لیے ۱۰۰ میرٹ اسکالرشپس دستیاب ہیں، جو بہترین تعلیمی کارکردگی والے اسٹوڈنٹس کو دی جاتی ہیں۔

پہلے سال کے لیے، یہ میرٹ اسکالرشپ ایک سال کے لیے ہوتی ہے اور اس میں پارشل سے لے کر فل ٹیوشن فیس کور کی جاتی ہے۔ یہ اسکالرشپ ہر اسکول کے لیے شارٹ لسٹ ہونے والے ٹاپ امیدواروں کو ان کے ہائی اکیڈمک کریڈنشلز کی بنیاد پر دی جاتی ہے۔
```

**Claim → sources:**

- لمز انڈرگریجویٹ اسٹوڈنٹس کو میرٹ اسکالرشپس دیتا ہے۔  
  ↳ [1] lums / BS Computer Science, Undergraduate Merit Scholarships, [4] lums / BS Computer Science, Financial Support at LUMS
- لمز میں انڈرگریجویٹ اسٹوڈنٹس کے لیے ۱۰۰ میرٹ اسکالرشپس دستیاب ہیں۔  
  ↳ [2] lums / BS Computer Science, Undergraduate Merit Scholarships, _First Year_, [4] lums / BS Computer Science, Financial Support at LUMS
- پہلے سال کے لیے یہ میرٹ اسکالرشپ ایک سال کے لیے ہوتی ہے اور اس میں پارشل سے لے کر فل ٹیوشن فیس کور ہوتی ہے۔  
  ↳ [2] lums / BS Computer Science, Undergraduate Merit Scholarships, _First Year_
- یہ اسکالرشپ ہر اسکول کے لیے شارٹ لسٹ ہونے والے ٹاپ امیدواروں کو ان کے اکیڈمک کریڈنشلز کی بنیاد پر دی جاتی ہے۔  
  ↳ [2] lums / BS Computer Science, Undergraduate Merit Scholarships, _First Year_

### U9 — حبیب یونیورسٹی میں کمپیوٹر سائنس کی فیس کتنی ہے؟
*Expect:* abstain — *Ground truth:* Habib fees not ingested -> abstain

*Probes:* Urdu faithful abstention

**Answer:**
```
مجھے فراہم کردہ معلومات میں حبیب یونیورسٹی کے کمپیوٹر سائنس پروگرام کی فیس کے بارے میں کوئی تفصیلات نہیں ملیں۔
```

**Claim → sources:**

- مجھے فراہم کردہ معلومات میں حبیب یونیورسٹی کے کمپیوٹر سائنس پروگرام کی فیس کے بارے میں کوئی تفصیلات نہیں ملیں۔  
  ↳ (none)

### U10 — نسٹ میں کمپیوٹر سائنس کے لیے کیا شرائط ہیں؟
*Expect:* abstain — *Ground truth:* NUST not ingested -> abstain

*Probes:* Urdu out-of-scope abstention

**Answer:**
```
مجھے افسوس ہے، لیکن دیے گئے ذرائع میں نسٹ میں کمپیوٹر سائنس کے داخلے کی شرائط کے بارے میں کوئی معلومات نہیں ہیں۔ میں صرف حبیب یونیورسٹی اور لمز کے کمپیوٹر سائنس پروگرامز کے بارے میں معلومات فراہم کر سکتی ہوں۔
```

**Claim → sources:** (no claims / abstained)

### U11 — کیا میں 75 پرسنٹ کے ساتھ لمز میں داخلہ لے سکوں گا؟
*Expect:* decline-predict — *Ground truth:* should not predict; redirect to requirements

*Probes:* Urdu counsellor boundary

**Answer:**
```
لیمز (LUMS) میں داخلہ ملنے کا فیصلہ صرف آپ کے مارکس پر نہیں ہوتا، بلکہ یہ کئی دوسرے عوامل پر بھی منحصر ہوتا ہے۔ اس لیے میں یہ نہیں بتا سکتی کہ آپ کا داخلہ ہوگا یا نہیں، لیکن میں آپ کو وہاں کی تعلیمی ضروریات ضرور بتا سکتی ہوں۔

لیمز کے مطابق، اگر آپ نے میٹرک اور ایف ایس سی (FSc) یا آئی سی ایس (ICS) مکمل کر لیا ہے یا ان کے فائنل رزلٹ کا انتظار کر رہے ہیں، تو داخلے کے لیے اہل ہونے کے لیے آپ کے میٹرک میں کم از کم ستر (70) پرسنٹ اور ایف ایس سی یا آئی سی ایس (پارٹ-1 یا مکمل) میں بھی کم از کم ستر (70) پرسنٹ مارکس ہونا ضروری ہیں۔ اس لحاظ سے پچھتر (75) پرسنٹ مارکس کے ساتھ آپ اس تعلیمی شرط کو پورا کرتے ہیں۔

اگر آپ امریکن ہائی اسکول ڈپلومہ کے طالب علم ہیں، تو اس کے لیے بھی ستر (70) پرسنٹ یا اس سے زیادہ، یا پھر بی (B) گریڈ پوائنٹ ایوریج ہونا لازمی ہے۔
```

**Claim → sources:**

- لیمز کے مطابق، اگر آپ نے میٹرک اور ایف ایس سی یا آئی سی ایس مکمل کر لیا ہے یا ان کے فائنل رزلٹ کا انتظار کر رہے ہیں، تو داخلے کے لیے اہل ہونے کے لیے آپ کے میٹرک میں کم از کم ستر (70) پرسنٹ اور ایف ایس سی یا آئی سی ایس (پارٹ-1 یا مکمل) میں بھی کم از کم ستر (70) پرسنٹ مارکس ہونا ضروری ہیں۔  
  ↳ [3] lums / BS Computer Science, Applicants with an Incomplete Higher-Secondary Degree, [4] lums / BS Computer Science, Applicants with Complete Higher-Secondary Degree
- اگر آپ امریکن ہائی اسکول ڈپلومہ کے طالب علم ہیں، تو اس کے لیے بھی ستر (70) پرسنٹ یا اس سے زیادہ، یا پھر بی (B) گریڈ پوائنٹ ایوریج ہونا لازمی ہے۔  
  ↳ [3] lums / BS Computer Science, Applicants with an Incomplete Higher-Secondary Degree, [4] lums / BS Computer Science, Applicants with Complete Higher-Secondary Degree

### U12 — کیا لمز کے لیے آئی بی سی سی ایکوئیولینس سرٹیفکیٹ ضروری ہے؟
*Expect:* answer — *Ground truth:* ہاں، اگر قابلیت Matric/FSc کے مساوی نہ ہو

*Probes:* Urdu IBCC equivalence

**Answer:**
```
لمز میں ان امیدواروں کو آئی بی سی سی سے ایکوئیولینس سرٹیفکیٹ حاصل کرنا ہوگا جن کی تعلیم میٹرک، ایف اے، آئی سی ایس، آئی کام، یا ایف ایس سی کے علاوہ کسی اور تعلیمی نظام سے مکمل ہوئی ہو یا وہ اس کے آخری سال میں ہوں۔ اس کے علاوہ، سید بابر علی سکول آف سائنس اینڈ انجینئرنگ میں اپلائی کرنے کے لیے سائنس سبجیکٹس میں آئی بی سی سی کا ایکوئیولینس سرٹیفکیٹ جمع کروانا ضروری ہے۔ کیمیکل اور الیکٹریکل انجینئرنگ کے لیے یہ ایکوئیولینس پری انجینئرنگ اسٹریم کے برابر ہونی چاہیے۔ ساتھ ہی، تمام امیدواروں کے لیے آئی بی سی سی اور ایچ ای سی جیسے اداروں سے اپنی تعلیمی اسناد کی تصدیق کروانا بھی لازمی ہے۔
```

**Claim → sources:**

- لمز میں ان امیدواروں کو آئی بی سی سی سے ایکوئیولینس سرٹیفکیٹ حاصل کرنا ہوگا جن کی تعلیم میٹرک، ایف اے، آئی سی ایس، آئی کام، یا ایف ایس سی کے علاوہ کسی اور تعلیمی نظام سے مکمل ہوئی ہو یا وہ اس کے آخری سال میں ہوں۔  
  ↳ [1] lums / BS Computer Science, Do I need to get an equivalence certificate from IBCC?
- سید بابر علی سکول آف سائنس اینڈ انجینئرنگ میں اپلائی کرنے کے لیے سائنس سبجیکٹس میں آئی بی سی سی کا ایکوئیولینس سرٹیفکیٹ جمع کروانا ضروری ہے۔  
  ↳ [2] lums / BS Computer Science, Applicants with Complete Higher-Secondary Degree, [3] lums / BS Computer Science, Applicants with an Incomplete Higher-Secondary Degree, [5] lums / BS Computer Science, External Scholarships, Applicants with Complete Higher-Secondary Degree
- کیمیکل اور الیکٹریکل انجینئرنگ کے لیے یہ ایکوئیولینس پری انجینئرنگ اسٹریم کے برابر ہونی چاہیے۔  
  ↳ [2] lums / BS Computer Science, Applicants with Complete Higher-Secondary Degree, [3] lums / BS Computer Science, Applicants with an Incomplete Higher-Secondary Degree, [5] lums / BS Computer Science, External Scholarships, Applicants with Complete Higher-Secondary Degree
- تمام امیدواروں کے لیے آئی بی سی سی اور ایچ ای سی جیسے اداروں سے اپنی تعلیمی اسناد کی تصدیق کروانا بھی لازمی ہے۔  
  ↳ [3] lums / BS Computer Science, Applicants with an Incomplete Higher-Secondary Degree, [4] lums / BS Computer Science, Applicants with Complete Higher-Secondary Degree
