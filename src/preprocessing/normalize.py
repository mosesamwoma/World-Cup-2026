# ============================================================
#  Team name & competition normalization
# ============================================================

TEAM_ALIASES = {
    "FR Germany":            "Germany",
    "West Germany":          "Germany",
    "Yugoslavia":            "Serbia",
    "Czechoslovakia":        "Czech Republic",
    "Soviet Union":          "Russia",
    "Dutch East Indies":     "Indonesia",
    "Zaire":                 "DR Congo",
    "Republic of Ireland":   "Ireland",
    "China PR":              "China",
    "Korea Republic":        "South Korea",
    "Korea DPR":             "North Korea",
    "USA":                   "United States",
    "Türkiye":               "Turkey",
    "North Macedonia":       "North Macedonia",
    "Cape Verde":            "Cape Verde Islands",
}

COMPETITION_MAP = {
    "FIFA World Cup":                        "FIFA World Cup",
    "FIFA World Cup qualification":          "WC Qualifier",
    "UEFA Euro":                             "Continental Championship",
    "Copa América":                          "Continental Championship",
    "Africa Cup of Nations":                 "Continental Championship",
    "AFC Asian Cup":                         "Continental Championship",
    "CONCACAF Gold Cup":                     "Continental Championship",
    "UEFA Nations League":                   "Nations League",
    "Confederations Cup":                    "Confederations Cup",
    "Friendly":                              "Friendly",
    "African Nations Championship":          "Continental Qualifier",
    "CONCACAF Nations League":               "Nations League",
    "CONMEBOL–UEFA Cup of Champions":        "Friendly",
}

CONFEDERATION_MAP = {
    "UEFA":     ["Germany", "France", "Spain", "England", "Portugal", "Italy",
                 "Netherlands", "Belgium", "Croatia", "Switzerland", "Turkey",
                 "Poland", "Serbia", "Denmark", "Austria", "Scotland", "Ukraine",
                 "Czech Republic", "Hungary", "Slovakia", "Slovenia", "Albania",
                 "Georgia", "North Macedonia", "Romania", "Wales"],
    "CONMEBOL": ["Brazil", "Argentina", "Uruguay", "Colombia", "Chile",
                 "Ecuador", "Paraguay", "Peru", "Bolivia", "Venezuela"],
    "CAF":      ["Morocco", "Senegal", "Nigeria", "Ghana", "Cameroon",
                 "Ivory Coast", "Egypt", "Mali", "South Africa", "Tunisia",
                 "Algeria", "DR Congo"],
    "AFC":      ["Japan", "South Korea", "Australia", "Iran", "Saudi Arabia",
                 "China", "Qatar", "Iraq", "Jordan", "Uzbekistan"],
    "CONCACAF": ["United States", "Mexico", "Canada", "Jamaica", "Panama",
                 "Costa Rica", "Honduras", "El Salvador", "Trinidad and Tobago",
                 "Haiti", "Guatemala"],
    "OFC":      ["New Zealand"],
}

# Reverse lookup: team → confederation
TEAM_CONFEDERATION = {
    team: conf
    for conf, teams in CONFEDERATION_MAP.items()
    for team in teams
}


def normalize_team(name: str) -> str:
    return TEAM_ALIASES.get(name.strip(), name.strip())


def normalize_competition(name: str) -> str:
    for key, val in COMPETITION_MAP.items():
        if key.lower() in name.lower():
            return val
    return "Friendly"


def get_confederation(team: str) -> str:
    return TEAM_CONFEDERATION.get(team, "Unknown")
