schema={
  "type": "object",
  "title": "Brochure Extraction Schema",
  "$schema": "http://json-schema.org/draft-07/schema#",
  "description": "Schema for extracting key project information in English language only. All extracted text values must be in English - translate from other languages when necessary.",
  "properties": {
    "projectName": {
      "type": "string",
      "title": "Project Name",
      "description": "The official name of the residential project in English. If found in another language, translate to English. If not found return \"Not Present\""
    },
    "projectAddress": {
      "type": "object",
      "title": "Project Address",
      "properties": {
        "Address": {
          "type": "string",
          "title": "Project Address",
          "description": "The full address or location of the project in English. If found in another language, translate to English. If not found return \"Not Present\"."
        },
        "City": {
          "type": "string",
          "title": "City",
          "description": "Extract the exact city name in English where the project is located, based on the brochure content. Only provide the city, not the locality or full address. If the city is not found in the document, return \"Not Present\"."
        },
        "Locality": {
          "type": "string",
          "title": "Locality",
          "description": "Extract the specific locality, neighborhood, or area in English within the city where the project is being developed. This should be more granular than the city name and must be directly mentioned in the brochure. If the locality is not found in the document, return \"Not Present\"."
        }
      },
      "required": ["Address", "City", "Locality"]
    },
    "builder": {
      "type": "object",
      "title": "Builder Information",
      "properties": {
        "name": {
          "type": "string",
          "title": "Builder Name",
          "description": "The name of the builder or developer in English. If found in another language, translate to English."
        },
        "BuilderWebsite": {
          "type": "string",
          "title": "Builder Website",
          "description": "Official website for the Project Builder"
        },
        "boundingBoxLTRB": {
          "type": "string",
          "title": "Builder Logo's Bounding Box (LTRB)",
          "description": "Bounding box coordinates for the builder logo image in LTRB format. The builder logo is usually a unique graphic or symbol representing the real estate developer or construction company. It often includes the company name, initials, or a distinctive icon. Make sure to choose the right logo as there are multiple occurence of the logo throughout the document choose the best one which represents the builder properly and should only be a logo without any other content in it.If not found return \"Not Present\"."
        },
        "pageNumber": {
          "type": "integer",
          "title": "Page Number",
          "description": "Page number where the builder logo image is present."
        },
        "Font": {
          "type": "string",
          "description": "The exact font type of the builder logo. Should be accurate"
        }
      },
      "description": "Details about the builder including name and logo image information. If not found return \"Not Present\"."
    },
    "floorplanConfigs": {
      "type": "array",
      "items": {
        "type": "object",
        "title": "Floorplan Configuration",
        "properties": {
          "bhkType": {
            "type": "string",
            "title": "BHK Type",
            "description": "Type of floorplan in English (e.g., 2 BHK, 3 BHK, etc.). If not present return it just as Floorplan"
          },
          "totalArea": {
            "type": "string",
            "title": "Total Area",
            "description": "Total area of the unit (if present). If not found return \"Not Present\"."
          },
          "carpetArea": {
            "type": "string",
            "title": "Carpet Area",
            "description": "Carpet area of the unit (if present). If not found return \"Not Present\"."
          },
          "superBuiltupArea": {
            "type": "string",
            "title": "Super Built-up Area",
            "description": "Super built-up area of the unit (if present). If not found return \"Not Present\"."
          },
          "builtupArea": {
            "type": "string",
            "title": "Built-up Area",
            "description": "Built-up area of the unit (if present). If not found return \"Not Present\"."
          },
          "saleableArea": {
            "type": "string",
            "title": "Saleable Area",
            "description": "Saleable area of the unit (if present)."
          },
          "boundingBoxLTRB": {
            "type": "string",
            "title": "Bounding Box (LTRB)",
            "description": "Bounding box coordinates for the image in LTRB format."
          },
          "pageNumber": {
            "type": "integer",
            "title": "Page Number",
            "description": "Page number where the image is present."
          }
        },
        "description": "Details for a specific floorplan type."
      },
      "title": "Floorplan Configurations",
      "description": "List of floorplan configurations (BHK types) with area details and associated image information. Should extract all floorplans and their respective bounding boxes accurately."
    },
    "amenities": {
      "type": "array",
      "items": {
        "type": "string",
        "title": "Amenity",
        "description": "A single amenity provided in the project in English."
      },
      "title": "Amenities",
      "description": "Extract amenities in English explicitly mentioned in the brochure that exactly match or closely correspond to entries in the predefined Reference List of Amenities. Only include exact matches or high-confidence partial matches (synonyms, minor wording variations). Do not include inferred or unlisted amenities. If none are found, return \"Not Present\".\n\nReference List includes: Anti-termite Treatment, Gated Community, Paved Compound, Earthquake Resistant, Internal Street Lights, Vastu Compliant, Grade A Building, Wheelchair Accessible, Feng Shui, Heli-Pad, Society Office, Well-Maintained Internal Roads, Permeable Pavement, Private Gardens/Balconies, Energy Efficient Lighting, Solar Lighting, Solar Water Heating, Solar Panel, Thermal Insulation, Creche/Day care, School, Temple, Community Hall, Library, Pet Park, Co-Working Spaces, Outdoor Event Spaces, Carrom, Chess, Dart Board, Billiards, Pool Table, Bowling Alley, Foosball, Air Hockey, Cricket Pitch, Football Ground, Table Tennis, Volley Ball Court, Lawn Tennis Court, Beach Volley Ball Court, Basketball Court, Skating Rink, Squash Court, Badminton Court, Rock Climbing Wall, Futsal, Video Gaming Room, Massage Room, Yoga/Meditation Area, Acupressure Park, Jacuzzi, Spa, Ayurvedic Centre, Aerobics Centre, Doctor on Call, Reflexology Park, Sauna, Clinic, Steam Room, Gymnasium, Pilates studios, Medical Centre, Landscape Garden, Open Space, Manicured Garden, Terrace Garden, Flower Garden, Fountain, Natural Pond, Pedestrian-Friendly Zones, Archery Range, Water Park/Slides, Community Garden/Urban Farming, Green Wall (Vertical Gardens), Forest Trail, Senior Citizen Sitout, Sit Out Area, Park, Cabana Sitting, EV Charging Stations, Car Parking, Music Room, Dance Studio, Art and Craft Studio, Club House, Multipurpose Hall, Children's Play Area, TOT - LOT, Sand Pit, Nature Trail, Swimming Pool, Multipurpose Court, Theatre, Golf Course, Banquet Hall, Sun Deck, Party Lawn, Gazebo, Barbecue, Amphitheatre, Jogging Track, Theater Home, Mini Theatre, Card Room, Indoor Games, Cycling Track, Theme Park, Gaming Zones, Wine Cellar, Art Gallery, Golf Putty, Golf Simulator, Fire NOC, CCTV Camera Security, Security Cabin, Fire Fighting Systems, Smoke Detectors, Gas Leak Detectors, 24x7 Security, Video Door Security, Biometric/Smart Card Access, Fire Alarm, Emergency Exits, Boom Barrier, Intercom Facilities, Signage and Road Markings, Car-Free Zones, Ambulance Service, Panic Buttons in Apartments, Emergency Evacuation Chairs, Fall Detection Systems in Bathrooms, Defibrillators in Common Areas, Rooftop Lounge, Bar/Chill-Out Lounge, Lounge, Cigar Lounge, Pergola, Reading Lounge, Business Lounge, Conference Room, Waiting Lounge, Food Court, Restaurant, Cafeteria, Smart Home Automation, Piped Gas, RO System, Wi-Fi Connectivity, Wi-Fi Zones in Common Areas, DTH Television, Laundry, Laundromat, Changing Area, Salon, Automated Car Wash, Car wash area, Milk Booth, Letter Box, Name Plates, Shopping Centre, Grocery Shop, Bus Shelter, Petrol Pump, Toilet for drivers, Concierge Service, Lift(s), Property Staff, Entrance Lobby, Intercom, ATM, Maintenance Staff, DG Availability, Escalators, Power Back up Lift, 24/7 Power Backup, Underground Electric Cabling, Centralized Air Conditioning, Power Substation, Air Purification Systems, Noise Insulation in Apartments, Plumber/Electrician on Call, Secretarial Services, Braille Signage, Waste Management, Garbage Disposal, Sewage Treatment Plant, Garbage Chute, Waste Segregation and Disposal, Recycling Facilities, Organic Waste Converter, Composting Facilities, 24/7 Water Supply, Water Softener Plant, Rain Water Harvesting, Borewell Water Supply, Municipal Water Supply, Smart Water Meters, Greywater Recycling, Low Flow Fixtures, Bioswales, Ground Water Recharging Systems, Water Treatment Plant."
    },
    "masterplanImage": {
      "type": "object",
      "title": "Masterplan Image",
      "properties": {
        "boundingBoxLTRB": {
          "type": "string",
          "title": "Bounding Box (LTRB)",
          "description": "Bounding box coordinates for the image in LTRB format. If not found return \"Not Present\"."
        },
        "pageNumber": {
          "type": "integer",
          "title": "Page Number",
          "description": "Page number where the image is present."
        }
      },
      "description": "Image id, bounding box (LTRB), and page number for the masterplan image. If not found return \"Not Present\"."
    },
    "locationMapImage": {
      "type": "object",
      "title": "Location Map Image",
      "properties": {
        "boundingBoxLTRB": {
          "type": "string",
          "title": "Bounding Box (LTRB)",
          "description": "Bounding box coordinates for the image in LTRB format. If not found return \"Not Present\"."
        },
        "pageNumber": {
          "type": "integer",
          "title": "Page Number",
          "description": "Page number where the image is present."
        }
      },
      "description": "Image id, bounding box (LTRB), and page number for the location map image. If not found return \"Not Present\"."
    },
    "interior_specification": {
      "type": "array",
      "items": {
        "type": "string",
        "title": "Specifications",
        "description": "Structural specifications of the buildings in English"
      },
      "title": "Interior Specifications of the Project Buildings.",
      "description": "Extract ONLY confirmed interior and structural specifications in English from project brochures that are explicitly stated as committed features. Follow these rules:\n\n1. Extract only specifications clearly mentioned in brochure sections describing interiors, finishes, or technical details, using definitive language (e.g., 'includes', 'comes with', 'features').\n2. Exclude general marketing claims, amenity descriptions (e.g., clubhouse, park), conditional or vague language ('may include', 'subject to availability'), and non-specific structural elements.\n3. Focus on specific materials, brands, technical or functional features (e.g., 'Italian marble flooring', 'modular kitchen', 'UPVC windows').\n4. Group related specifications logically; create new categories if needed.\n5. If no confirmed specifications found, return 'Not Present'.\n6. Provide only specific, verified features without extra explanation in English."
    },
    "amenitiesImages": {
      "type": "array",
      "items": {
        "type": "object",
        "title": "Amenity Image",
        "properties": {
          "amenityLabel": {
            "type": "string",
            "title": "Amenity Label",
            "description": "Label or name of the amenity depicted in the image in English. If found in another language, translate to English. If not found return \"Not Present\"."
          },
          "boundingBoxLTRB": {
            "type": "string",
            "title": "Bounding Box (LTRB)",
            "description": "Bounding box coordinates for the image in LTRB format. If not found return \"Not Present\"."
          },
          "pageNumber": {
            "type": "integer",
            "title": "Page Number",
            "description": "Page number where the image is present."
          }
        },
        "description": "Image and label for a specific amenity."
      },
      "title": "Amenities Images",
      "description": "An array of images visually representing amenities. Only include images that clearly depict the actual amenity (e.g., photos or detailed illustrations of the scene), not simple icons, decorative graphics, or text-only designs. Strictly exclude images that only show text, graphic symbols, or design labels. If no such images exist, return Not Present"
    },
    "area": {
      "type": "object",
      "title": "Area Information",
      "properties": {
        "project_area": {
          "type": "string",
          "title": "Project Area",
          "description": "Total physical area within the project boundaries, including all land designated for construction, development, and related activities. Expressed in units like sq.ft, sq.m, or acres."
        },
        "open_area": {
          "type": "string",
          "title": "Open Area of the Project",
          "description": "Open area within the project, represented either as a percentage of the total site or as a physical area (e.g., sq.ft or acres). It includes all non-built-up spaces meant for common use."
        },
        "green_area": {
          "type": "string",
          "title": "Green Area of the Project",
          "description": "Green area within the project, expressed either as a percentage of the total site or as a physical area (e.g., in sq.ft or acres)."
        }
      }
    },
    "tower_count": {
      "type": "integer",
      "title": "Tower Count",
      "description": "Tower Count in project refers to the total number of vertical structures or towers planned or constructed within the project site"
    },
    "tower_names": {
      "type": "array",
      "items": {
        "type": "string",
        "title": "Tower Name",
        "description": "Name or label of an individual tower in English."
      },
      "title": "Tower Names",
      "description": "List of tower names found in the document in English. If names are in another language, translate to English. Ensure the number of names matches the tower_count if both are present."
    },
    "unit_count": {
      "type": "string",
      "title": "Unit Count",
      "description": "Total number of individual units (e.g., flats, villas) available for ownership or occupancy in the project."
    },
    "rera": {
      "type": "string",
      "title": "RERA Number",
      "description": "RERA number of the project the brochure is about. If not found return \"Not Present\"."
    },
    "location_highlights": {
      "type": "array",
      "items": {
        "type": "object",
        "title": "Location Highlight",
        "properties": {
          "category": {
            "type": "string",
            "title": "Category",
            "description": "The predefined category of the location using fuzzy matching to these categories: ATM, Airport, Amusement Park, Bank, Beach, Bus Stop, Club, Connectivity, Day Care Center, Golf Course, Highway, Hills, Hospital, Hotel, Metro Station, Nursery / Pre-school, Office Complex, Old Age Home, Park, Parking, Petrol Pump, Pharmacy, Railway Station, Religious Place, School, Shopping Center / Market / Mall, Stadium, University, Others"
          },
          "location_name": {
            "type": "string",
            "title": "Location Name",
            "description": "The exact name of the location in English as mentioned in the document. If in another language, provide English translation."
          },
          "distance": {
            "type": "string",
            "title": "Distance",
            "description": "The exact numerical distance or time metric (e.g., '500 meter', '8 Km', '15 minutes') as mentioned in the document."
          }
        },
        "description": "A single location highlight with category, location name, and distance."
      },
      "title": "Location Highlights",
      "description": "A list of major locations with explicit numerical distance metrics, each categorized and including the exact location name and distance as found in the document. If not found return \"Not Present\"."
    }
  }
}
