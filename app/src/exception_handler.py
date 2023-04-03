from fastapi import Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

def get_error_response(request, exc) -> dict:
    error_response = {
        'error': True,
        'message': str(exc)
    }

    return error_response

async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Handling error in validating requests
    """
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=get_error_response(request, exc)
    )

async def python_exception_handler(request: Request, exc: Exception):
    """
    Handling any internal error
    """

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=get_error_response(request, exc)
    )
