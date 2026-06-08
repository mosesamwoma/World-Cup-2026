# ============================================================
#  Team name & competition normalization
# ============================================================

TEAM_ALIASES = {
    # Historical names → current FIFA names
    "FR Germany":               "Germany",
    "West Germany":             "Germany",
    "Yugoslavia":               "Serbia",
    "Czechoslovakia":           "Czech Republic",
    "Soviet Union":             "Russia",
    "Dutch East Indies":        "Indonesia",
    "Zaire":                    "DR Congo",
    "Republic of Ireland":      "Ireland",
    "China PR":                 "China",
    "Korea Republic":           "South Korea",
    "Korea DPR":                "North Korea",
    "USA":                      "United States",
    "Türkiye":                  "Turkey",
    # FIXED: was converting Cape Verde → Cape Verde Islands
    # but fixtures use "Cape Verde" — keep that as the canonical name
    "Cape Verde Islands":       "Cape Verde",
}

COMPETITION_MAP = {
    "FIFA World Cup":                        "FIFA World Cup",
    "FIFA World Cup qualification":          "WC Qualifier",
    "UEFA Euro":                             "Continental Championship",
    "UEFA European Championship":            "Continental Championship",
    "Copa América":                          "Continental Championship",
    "Africa Cup of Nations":                 "Continental Championship",
    "African Cup of Nations":                "Continental Championship",
    "AFC Asian Cup":                         "Continental Championship",
    "CONCACAF Gold Cup":                     "Continental Championship",
    "UEFA Nations League":                   "Nations League",
    "CONCACAF Nations League":               "Nations League",
    "Confederations Cup":                    "Confederations Cup",
    "African Nations Championship":          "Continental Qualifier",
    "CONMEBOL–UEFA Cup of Champions":        "Friendly",
    "Friendly":                              "Friendly",
}

CONFEDERATION_MAP = {
    "UEFA": [
        "Germany", "France", "Spain", "England", "Portugal", "Italy",
        "Netherlands", "Belgium", "Croatia", "Switzerland", "Turkey",
        "Poland", "Serbia", "Denmark", "Austria", "Scotland", "Ukraine",
        "Czech Republic", "Hungary", "Slovakia", "Slovenia", "Albania",
        "Georgia", "North Macedonia", "Romania", "Wales",
        "Norway", "Sweden", "Bosnia and Herzegovina",
    ],
    "CONMEBOL": [
        "Brazil", "Argentina", "Uruguay", "Colombia", "Chile",
        "Ecuador", "Paraguay", "Peru", "Bolivia", "Venezuela",
    ],
    "CAF": [
        "Morocco", "Senegal", "Nigeria", "Ghana", "Cameroon",
        "Ivory Coast", "Egypt", "Mali", "South Africa", "Tunisia",
        "Algeria", "DR Congo", "Cape Verde",
    ],
    "AFC": [
        "Japan", "South Korea", "Australia", "Iran", "Saudi Arabia",
        "China", "Qatar", "Iraq", "Jordan", "Uzbekistan",
    ],
    "CONCACAF": [
        "United States", "Mexico", "Canada", "Jamaica", "Panama",
        "Costa Rica", "Honduras", "El Salvador", "Trinidad and Tobago",
        "Haiti", "Guatemala", "Curaçao",
    ],
    "OFC": [
        "New Zealand",
    ],
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