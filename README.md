# Evacuation Patterns and Socioeconomic Stratification in the Context of Wildfires

## Overview
This repository contains the code and data analysis related to the research paper _"Evacuation Patterns and Socioeconomic Stratification in the Context of Wildfires"_. The study investigates the evacuation behaviours during the wildfires in Valparaíso, Chile, in February 2024, focusing on the role of socioeconomic factors.

## Authors
- Timur Naushirvanov
- Erick Elejalde
- Kyriaki Kalimeri
- Elisa Omodei
- Márton Karsai
- Leo Ferres

## Abstract
Climate change is altering the frequency and intensity of wildfires, leading to increased evacuation events that disrupt human mobility and socioeconomic structures. These disruptions affect access to resources, employment, and housing, amplifying existing vulnerabilities within communities. Understanding the interplay between climate change, wildfires, evacuation patterns, and socioeconomic factors is crucial for developing effective mitigation and adaptation strategies. To contribute to this challenge, we use high-definition mobile phone records to analyse evacuation patterns during the wildfires in Valparaíso, Chile, that took place between February 2-3, 2024. This data allows us to track the movements of individuals in the disaster area, providing insight into how people respond to large-scale evacuations in the context of severe wildfires. We apply a causal inference approach that combines regression discontinuity and difference-in-differences methodologies to observe evacuation behaviours during wildfires, with a focus on socioeconomic stratification. This approach allows us to isolate the impact of the wildfires on different socioeconomic groups by comparing the evacuation patterns of affected populations before and after the event, while accounting for underlying trends and discontinuities at the threshold of the disaster. We find that many people spent nights away from home, with those in the lowest socioeconomic segment stayed away the longest. In general, people reduced their travel distance during the evacuation, and the lowest socioeconomic group moved the least. Initially, movements became more random, as people sought refuge in a rush, but eventually gravitated towards areas with similar socioeconomic status. Our results show that socioeconomic differences play a role in evacuation dynamics, providing useful insights for response planning.

Key findings include:
- Lower socioeconomic groups tend to stay away from their homes longer and reduce travel more significantly during evacuations.
- Initial movements were chaotic, but individuals gradually gravitated toward areas with similar socioeconomic conditions.

For more information, please refer to the pre-print [here](https://ar5iv.org/abs/2410.06017).

## Repository Structure

- `/scripts/`: Python scripts for data preprocessing, analysis (including regression discontinuity and difference-in-differences models), and visualisations.
- `/visuals_created/`: Plots and figures generated from the analysis.
- `/notebooks/`: Jupyter notebooks with plots and analysis comparing Telefónica data with Facebook Disaster Maps (FBDM).

## Installation
To run the code in this repository, ensure you have the following dependencies installed:

```bash
pip install -r requirements.txt
```


## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact
If you have any questions or issues with the code, feel free to reach out to the authors.

