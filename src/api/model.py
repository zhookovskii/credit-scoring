from pydantic import BaseModel


class Applicant(BaseModel):
    AMT_INCOME_TOTAL: float
    AMT_CREDIT: float
    AMT_ANNUITY: float
    AMT_GOODS_PRICE: float
    AGE_YEARS: float
    EMPLOYMENT_YEARS: float
    CODE_GENDER: str
    NAME_FAMILY_STATUS: str
    NAME_EDUCATION_TYPE: str
    NAME_INCOME_TYPE: str
    OCCUPATION_TYPE: str
    NAME_HOUSING_TYPE: str
    FLAG_OWN_CAR: str
    FLAG_OWN_REALTY: str
