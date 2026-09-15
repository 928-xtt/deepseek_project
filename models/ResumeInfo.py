from pydantic import BaseModel, Field, field_validator

class ResumeInfo(BaseModel):
    """简历提取结果的结构定义"""
    name: str = Field(..., description="姓名")
    years: int = Field(..., ge=0, le=50, description="工作年限")  # ge/le ≈ @Min/@Max
    skills: list[str] = Field(..., min_length=1, description="技能列表")
    email: str | None = None  # 可选字段

    # 自定义校验器 ≈ 自定义 @Constraint
    @field_validator("email")
    @classmethod
    def email_must_contain_at(cls, v):
        if v is not None and "@" not in v:
            raise ValueError("email 必须包含 @")
        return v
