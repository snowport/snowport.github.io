
# The Water Confidence Index (WCI): Its Development and Construction
*April 26, 2021*

![](https://upload.wikimedia.org/wikipedia/commons/9/96/Iowa_-_American_Water_-_Davenport_Water_Tower_%2824259031639%29.jpg)


## Objective
To develop an environmental composite index—**Water Confidence Index (WCI)**—that measures and ranks the performance of large U.S. public water systems (PWS) based on **environmental compliance** and **truthful reporting**.


## Motivation

- No existing index ranks U.S. water utilities by environmental health performance.
- Infrastructure challenges include underfunding, aging pipes, cybersecurity threats, and increasing violations.
- The index supports transparency, prioritization for funding, and accountability in public water infrastructure.


## WCI Structure

WCI = Average of:
- **Environmental Compliance (EC)**:
  
  - Health-based violations (40%)
  - Serious violator trends (40%)
  - Violations per site visit (20%)

- **Truthful Reporting (TR)**:
  
  - Public notice + monitoring violations (40%)
  - Enforcement-to-violation ratio (60%)

**Normalization:** Min-max method used to scale inputs.


## Data Sources

- **EPA ECHO / SDWIS datasets** (2011–2020)
- **Filtered to large & very large PWS** (serving >10,000 people)
- **Excludes**: Small systems, Tribal systems, Territories
- **Population data**: U.S. Census Bureau estimates


## Methodology

- Indicators derived from violations, enforcements, and compliance trends.
- Weighted arithmetic calculations yield EC and TR scores per state.
- Combined into a 0–1 **WCI score**, where **lower = better**.


## Results

**Top-performing states** (lowest WCI scores):
     
| Rank | State         | 
|------|---------------|
| 1    | Indiana --> (0.03) |     
| 2    | North Dakota  |         
| 3    | Minnesota     |         
| 4    | South Dakota  |         
| 5    | Michigan      |         


**Lowest-performing states** (highest WCI scores):
    
| Rank | State       | 
|------|-------------|
| 46   | Arizona     |       
| 47   | Texas       |       
| 48   | California  |       
| 49   | Mississippi |       
| 50   | Idaho   --> (0.86)    | 

**Top EPA Region:** Region 5 (Midwest)

**Worst EPA Region:** Region 10 (Northwest)

## Sensitivity Analyses

- **SA-1:** Reweighted EC to focus on health violations → minor rank shifts.
- **SA-2:** Reweighted TR to focus on public health violations → larger shifts for lower-ranking states.
- **SA-3:** Equal weighting of all inputs → stable top/bottom rankings; mid-range states more sensitive.


## Limitations & Future Directions

- Does not include small PWS or private wells.
- TR score may overemphasize certain violations due to uneven frequency.
- No economic or health outcome data included yet (e.g., cost of repair, hospital visits).
- Future enhancements:
    - Add third component for public notification.
    - Link violations to health impacts or cost estimates.
    - Visualize results by EPA Region for policy impact.

## Conclusion

- The WCI provides a **transparent, data-driven ranking** of public water utilities based on SDWA compliance.
- It can inform **public awareness**, **EPA oversight**, and **infrastructure funding allocation**.
- It highlights the power of composite indicators in environmental decision-making.

---

Full report available [here](pdfs/Water_Confidence_Index.pdf).

---