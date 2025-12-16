# Constitution of India - Entity Types Reference

This document provides reference information for the Constitutional entity types used in Apna Sanvidhan.

## Entity Types in Apna Sanvidhan

### 1. **PERSON**
Constitutional framers, historical figures, officials, judges mentioned in the Constitution.

**Examples:**
- Dr. B.R. Ambedkar (Principal Architect)
- Jawaharlal Nehru
- Members of Constituent Assembly
- Chief Justice of India
- President of India

### 2. **ORGANIZATION**
Governmental bodies, institutions, Parliament, Courts, Commissions mentioned in the Constitution.

**Examples:**
- Parliament of India
- Lok Sabha
- Rajya Sabha
- Supreme Court of India
- Election Commission of India
- State Governments
- Union Public Service Commission
- Union Territories

### 3. **LOCATION**
States, territories, geographical divisions of India mentioned in the Constitution.

**Examples:**
- States of India (Maharashtra, Delhi, etc.)
- Union Territories (Delhi, Puducherry, etc.)
- National Capital Territory
- Scheduled Areas
- Scheduled Tribes Areas

### 4. **ARTICLE/SECTION** ⭐ (Constitution-Specific)
Constitutional articles, sections, schedules, and constitutional provisions.

**Examples:**
- Article 15 (Right to Equality)
- Article 19 (Freedom of Speech)
- Article 21 (Right to Life)
- Article 44 (Uniform Civil Code)
- Schedule 1 (States and Union Territories)
- Schedule 8 (Languages)
- Part III (Fundamental Rights)
- Part IV (Directive Principles)

### 5. **FUNDAMENTAL_RIGHT** ⭐ (Constitution-Specific)
Rights guaranteed under Part III of the Constitution.

**Examples:**
- Equality before law (Article 14)
- Freedom of speech and expression (Article 19)
- Right to life and personal liberty (Article 21)
- Right to constitutional remedies (Article 32)
- Freedom of religion (Article 25-28)
- Right to education (Article 21A)
- Right to information (derived from Article 19)

### 6. **DIRECTIVE_PRINCIPLE** ⭐ (Constitution-Specific)
Constitutional directives mentioned in Part IV for State policy guidance.

**Examples:**
- State to ensure adequate means of livelihood (Article 39)
- Right to work (Article 41)
- Right to education (Article 45)
- Promotion of educational and economic interests of SCs/STs (Article 46)
- Living wage and conditions of work (Article 43)
- Uniform civil code (Article 44)
- Child labor prohibition (Article 39)

### 7. **DUTY** ⭐ (Constitution-Specific)
Fundamental Duties or statutory duties mentioned in the Constitution.

**Examples:**
- Respect the Constitution and the flag (Article 51A(a))
- Cherish and follow ideals of freedom struggle (Article 51A(b))
- Uphold sovereignty and integrity of India (Article 51A(c))
- Defend the country (Article 51A(d))
- Promote harmony and spirit of common brotherhood (Article 51A(e))
- Preserve cultural heritage (Article 51A(f))
- Protect environment and forests (Article 51A(g))
- Develop scientific temperament (Article 51A(h))
- Safeguard public property (Article 51A(i))
- Abjure violence (Article 51A(j))
- Strive for excellence (Article 51A(k))

### 8. **CONCEPT**
Abstract constitutional principles, doctrines, and fundamental concepts.

**Examples:**
- Sovereignty
- Democracy
- Secularism
- Federal structure
- Judicial review
- Separation of powers
- Rule of law
- Constitutionalism
- Fundamental rights doctrine
- Separation of church and state

### 9. **DATE**
Important dates, years, constitutional amendments, and historical events.

**Examples:**
- January 26, 1950 (Constitution enforcement date)
- November 26, 1949 (Constitution adoption date)
- 1952 (First General Elections)
- 2016 (73rd Amendment - Local governance)
- Constitutional amendments (1st, 44th, 101st, etc.)
- Special dates mentioned in the Constitution

---

## Example Entity Extraction

### Example Text:
"Article 15 prohibits discrimination on grounds of religion, race, caste, sex or place of birth. This Fundamental Right is a cornerstone of the Indian Constitution adopted by the Constituent Assembly on November 26, 1949, and enforced on January 26, 1950, under the presidency of Dr. Rajendra Prasad."

### Extracted Entities:

| Entity | Type | Description |
|--------|------|-------------|
| Article 15 | ARTICLE/SECTION | Constitutional provision on equality |
| Discrimination | CONCEPT | Constitutional principle |
| Religion, race, caste, sex | CONCEPT | Prohibited grounds |
| Fundamental Right | FUNDAMENTAL_RIGHT | Category of rights |
| Indian Constitution | CONCEPT | Foundational law |
| Constituent Assembly | ORGANIZATION | Body that framed Constitution |
| November 26, 1949 | DATE | Adoption date |
| January 26, 1950 | DATE | Enforcement date |
| Dr. Rajendra Prasad | PERSON | First President |

---

## Constitutional Structure for Context

### Parts of the Constitution

- **Preamble**: Objects and aims
- **Part I**: The Union and Its Territory
- **Part II**: Citizenship
- **Part III**: Fundamental Rights (Articles 12-35)
- **Part IV**: Directive Principles of State Policy (Articles 36-51)
- **Part IVA**: Fundamental Duties (Article 51A)
- **Part V**: The Union
- **Part VI**: The States
- **Part VII**: (Repealed)
- **Part VIII**: The Union Territories
- **Part IX**: The Panchayats
- **Part IX-A**: The Municipalities
- **Part X**: Scheduled and Tribal Areas
- **Part XI**: Relations between the Union and the States
- **Part XII**: Finance, Property, Contracts and Suits
- **Part XIII**: Trade, Commerce, and Intercourse within the Territory of India
- **Part XIV**: Services under the Union and the States
- **Part XIV-A**: Tribunals
- **Part XV**: Elections
- **Part XVI**: Special Provisions relating to Certain Classes
- **Part XVII**: Languages
- **Part XVIII**: Emergency Provisions
- **Part XIX**: Miscellaneous
- **Part XX**: Amendment of the Constitution
- **Part XXI**: Temporary and Transitional Provisions

### Schedules

- **Schedule 1**: States and Union Territories
- **Schedule 2**: Emoluments and allowances
- **Schedule 3**: Oaths and affirmations
- **Schedule 4**: Allocation of seats in Rajya Sabha
- **Schedule 5**: Provisions relating to Scheduled and Tribal Areas
- **Schedule 6**: Provisions relating to Assam, Meghalaya, Tripura and Mizoram
- **Schedule 7**: Division of legislative powers
- **Schedule 8**: Languages
- **Schedule 9**: Validation of certain Acts and Regulations (Land Reforms, etc.)
- **Schedule 10**: Disqualification of Members of Parliament and State Legislatures
- **Schedule 11**: Powers of Panchayats
- **Schedule 12**: Powers of Municipalities

---

## Tips for Query Success

When querying Apna Sanvidhan about the Constitution:

1. **Use Article Numbers**: "What does Article 14 say?" (more accurate than "What about equality?")
2. **Reference Fundamental Rights**: "What are the Fundamental Rights?" (recognized as FUNDAMENTAL_RIGHT entity)
3. **Mention Directive Principles**: "What Directive Principles relate to education?"
4. **Ask about Duties**: "What are the Fundamental Duties?" (Part IVA)
5. **Include Dates**: "When did the Constitution come into effect?" (recognized as DATE entity)
6. **Name Organizations**: "What is the role of Parliament?" (recognized as ORGANIZATION)
7. **Reference Concepts**: "What does secularism mean in the Constitution?"

---

## Example Queries for Apna Sanvidhan

### Local Search (Entity-based) Queries:
- "What does Article 19 protect?"
- "What is the role of the Election Commission?"
- "Explain Article 21 - Right to Life"
- "What Fundamental Duties apply to citizens?"
- "What is mentioned in Schedule 1?"

### Global Search (Community-based) Queries:
- "Explain the concept of secularism in the Constitution"
- "What are the principles of democracy in the Indian Constitution?"
- "How does the Constitution protect minorities?"
- "What is the federal structure of India?"

### Hybrid Search Queries:
- "How do Articles on Fundamental Rights relate to Directive Principles?"
- "What is the constitutional framework for ensuring equality and justice?"
- "How does the Constitution balance individual rights with social welfare?"

---

## Notes

- This is a living reference document
- Entity types are optimized for Constitutional analysis
- The system can handle multiple entity types in a single query
- Relationship extraction identifies connections between constitutional provisions
- Custom entity types ensure constitutional context is properly understood

---

**Last Updated**: December 2024
**System**: Apna Sanvidhan - Constitution of India SemRAG
