from datetime import date, datetime


def calculate_age_from_birthdate(birthdate_value, current_date=None) -> int:
    if isinstance(birthdate_value, datetime):
        birthdate = birthdate_value.date()
    elif isinstance(birthdate_value, date):
        birthdate = birthdate_value
    else:
        birthdate = datetime.fromisoformat(str(birthdate_value)).date()

    if current_date is None:
        today = datetime.now().date()
    elif isinstance(current_date, datetime):
        today = current_date.date()
    elif isinstance(current_date, date):
        today = current_date
    else:
        today = datetime.fromisoformat(str(current_date)).date()

    return today.year - birthdate.year - ((today.month, today.day) < (birthdate.month, birthdate.day))
