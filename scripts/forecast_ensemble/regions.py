"""Disjoint, exhaustive country->region map for LORO. Any country not listed -> 'other'
(stays in training, never a fold). Run directly to audit coverage against the panel."""
A = ["Angola","Benin","Botswana","Burkina Faso","Burundi","Cameroon","Cape Verde","Central African Republic",
"Chad","Comoros","Congo","Republic of Congo","Democratic Republic of Congo","Ivory Coast","Djibouti",
"Equatorial Guinea","Eritrea","eSwatini","Kingdom of eSwatini (Swaziland)","Ethiopia","Gabon","Gambia","Ghana",
"Guinea","Guinea-Bissau","Kenya","Lesotho","Liberia","Madagascar","Madagascar (Malagasy)","Malawi","Mali",
"Mauritania","Mauritius","Mayotte","Mozambique","Namibia","Niger","Nigeria","Reunion","Rwanda",
"Saint Helena, Ascension and Tristan da Cunha","Sao Tome and Principe","Senegal","Seychelles","Sierra Leone",
"Somalia","South Africa","South Sudan","Sudan","Tanzania","Togo","Uganda","Zambia","Zimbabwe","Zimbabwe (Rhodesia)"]
M = ["Algeria","Bahrain","Egypt","Iran","Iraq","Israel","Jordan","Kuwait","Lebanon","Libya","Morocco","Oman",
"Qatar","Saudi Arabia","Syria","Tunisia","Turkey","United Arab Emirates","Yemen","Yemen (North Yemen)",
"Western Sahara","Palestine"]
E = ["Albania","Andorra","Austria","Belarus","Belgium","Bosnia-Herzegovina","Bulgaria","Croatia","Cyprus",
"Czech Republic","Denmark","Estonia","Faroe Islands","Finland","France","Georgia","Germany","Gibraltar","Greece",
"Greenland","Hungary","Iceland","Ireland","Isle of Man","Italy","Latvia","Liechtenstein","Lithuania","Luxembourg",
"Malta","Moldova","Monaco","Montenegro","Netherlands","North Macedonia","Norway","Poland","Portugal","Romania",
"Russia","Russia (Soviet Union)","San Marino","Serbia","Serbia (Yugoslavia)","Slovakia","Slovenia","Spain",
"Sweden","Switzerland","Ukraine","United Kingdom","Armenia","Azerbaijan","Bailiwick of Guernsey",
"Bailiwick of Jersey","Akrotiri and Dhekelia","Vatican City"]
AM = ["United States","United States of America","Canada","Mexico","Guatemala","Belize","El Salvador","Honduras",
"Nicaragua","Costa Rica","Panama","Cuba","Haiti","Dominican Republic","Jamaica","Bahamas","Puerto Rico",
"Trinidad and Tobago","Barbados","Saint Lucia","Grenada","Dominica","Antigua and Barbuda",
"Saint Vincent and the Grenadines","Saint Kitts and Nevis","Aruba","Curacao","Sint Maarten",
"Caribbean Netherlands","Saint-Barthelemy","Guadeloupe","Martinique","Cayman Islands","Turks and Caicos Islands",
"British Virgin Islands","Virgin Islands, U.S.","Montserrat","Bermuda","Anguilla","Saint Pierre and Miquelon",
"Colombia","Venezuela","Ecuador","Peru","Bolivia","Brazil","Chile","Argentina","Uruguay","Paraguay","Guyana",
"Suriname","French Guiana","Falkland Islands"]
S = ["India","Pakistan","Bangladesh","Nepal","Sri Lanka","Afghanistan","Bhutan","Maldives","Kazakhstan",
"Kyrgyzstan","Tajikistan","Turkmenistan","Uzbekistan"]
P = ["China","Japan","North Korea","South Korea","Taiwan","Mongolia","Vietnam","Laos","Cambodia",
"Cambodia (Kampuchea)","Thailand","Myanmar","Malaysia","Singapore","Indonesia","Philippines","Brunei",
"Timor-Leste","East Timor","Papua New Guinea","Australia","New Zealand","Fiji","Solomon Islands","Vanuatu",
"New Caledonia","Samoa","American Samoa","Tonga","Tuvalu","Kiribati","Nauru","Palau","Marshall Islands",
"Micronesia","Guam","Northern Mariana Islands","Cook Islands","Niue","Tokelau","French Polynesia",
"Wallis and Futuna","Christmas Island"]
REGION_OF = {}
for lst, rg in [(A,"africa"),(M,"mena"),(E,"europe"),(AM,"americas"),(S,"south_asia"),(P,"asia_pacific")]:
    for c in lst:
        assert c not in REGION_OF, f"DUPLICATE country in two regions: {c}"
        REGION_OF[c] = rg
REGIONS = ["africa","mena","europe","americas","south_asia","asia_pacific"]
def region_of(c): return REGION_OF.get(c, "other")

if __name__ == "__main__":
    import pandas as pd
    R = "/storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast"
    d = pd.read_parquet(f"{R}/data_intermediate/baseline_features.parquet",
                        columns=["country","violence_next_month"])
    d["region"] = d["country"].map(region_of)
    g = d.groupby("region").agg(rows=("country","size"), viol=("violence_next_month","sum"))
    g["pct_rows"] = (100*g["rows"]/len(d)).round(2)
    print(g.sort_values("rows", ascending=False).to_string(), flush=True)
    oth = sorted(d.loc[d.region=="other","country"].unique())
    ov = d.loc[d.region=="other","violence_next_month"].sum()
    print(f"\nUNMAPPED ('other'): {len(oth)} countries, {ov} violent cell-months", flush=True)
    print(oth, flush=True)
