"""Data models for TNSIM API."""

from .requests import (
    CreateInfiniteSetRequest,
    ZeroSumOperationRequest,
    FindCompensatingSetRequest,
    ValidateZeroSumRequest,
    BatchOperationRequest
)
from .responses import (
    InfiniteSetResponse,
    ZeroSumOperationResponse,
    CompensatingSetResponse,
    ValidationResponse,
    BatchOperationResponse,
    ErrorResponse
)

__all__ = [
    # Requests
    'CreateInfiniteSetRequest',
    'ZeroSumOperationRequest',
    'FindCompensatingSetRequest',
    'ValidateZeroSumRequest',
    'BatchOperationRequest',
    # Responses
    'InfiniteSetResponse',
    'ZeroSumOperationResponse',
    'CompensatingSetResponse',
    'ValidationResponse',
    'BatchOperationResponse',
    'ErrorResponse'
]