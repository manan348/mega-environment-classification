# District Boundaries

Place your Punjab district boundaries file here as `punjab_districts.geojson` or `punjab_districts.shp`.

## Required Format

The file must contain:
- **36 features** (one per Punjab district)
- **Polygon/MultiPolygon** geometries
- A **district name column** (e.g., `NAME_2`, `DISTRICT`, `district_name`)

## Expected Districts (36)

1. Attock
2. Bahawalnagar
3. Bahawalpur
4. Bhakkar
5. Chakwal
6. Chiniot
7. Dera Ghazi Khan
8. Faisalabad
9. Gujranwala
10. Gujrat
11. Hafizabad
12. Jhang
13. Jhelum
14. Kasur
15. Khanewal
16. Khushab
17. Lahore
18. Layyah
19. Lodhran
20. Mandi Bahauddin
21. Mianwali
22. Multan
23. Muzaffargarh
24. Nankana Sahib
25. Narowal
26. Okara
27. Pakpattan
28. Rahim Yar Khan
29. Rajanpur
30. Rawalpindi
31. Sahiwal
32. Sargodha
33. Sheikhupura
34. Sialkot
35. Toba Tek Singh
36. Vehari

## Sources

- **GADM**: https://gadm.org/download_country.html (Pakistan, level 2)
- **Pakistan Bureau of Statistics**: Official district boundaries
- **Your Fasal Akhbar project**: You likely already have this

## Quick GeoJSON Template

```json
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "properties": {"NAME_2": "Attock"},
      "geometry": {"type": "Polygon", "coordinates": [[[...]]]}
    },
    ...
  ]
}
```

The pipeline auto-detects the district name column (looks for `NAME`, `DISTRICT`, or `district` in column names).