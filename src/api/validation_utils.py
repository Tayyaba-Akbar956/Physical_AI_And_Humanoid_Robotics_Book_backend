from typing import Dict, Any, Optional
from pydantic import BaseModel, Field, field_validator
from enum import Enum
import re


class QueryType(str, Enum):
    GENERAL = "general"
    TEXT_SELECTION = "text_selection"
    FOLLOW_UP = "follow_up"


class ChatQueryRequest(BaseModel):
    """
    Enhanced request model for chat queries with comprehensive validation
    """
    session_id: Optional[str] = Field(None, description="Session ID, if continuing conversation")
    message: str = Field(
        ..., 
        min_length=1, 
        max_length=2000,
        description="The user's query message"
    )
    module_context: Optional[str] = Field(
        None,
        description="Current module context for prioritization"
    )
    selected_text: Optional[str] = Field(
        None,
        min_length=20,
        max_length=10000,
        description="Text selected by user if this is a text selection query"
    )
    query_type: Optional[QueryType] = Field(
        QueryType.GENERAL,
        description="Type of query being made"
    )
    student_context: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional context about the student"
    )

    @field_validator('session_id')
    @classmethod
    def validate_session_id(cls, v):
        if v is not None:
            # Basic UUID validation - in reality we'd check if it's a valid UUID format
            import uuid
            try:
                uuid.UUID(v)
            except ValueError:
                raise ValueError('session_id must be a valid UUID')
        return v

    @field_validator('message')
    @classmethod
    def validate_message(cls, v):
        if not v or not v.strip():
            raise ValueError('Message cannot be empty or just whitespace')
        if len(v.strip()) < 3:
            raise ValueError('Message should be at least 3 characters long')
        return v.strip()

    @field_validator('module_context')
    @classmethod
    def validate_module_context(cls, v):
        if v is not None:
            # Basic validation of module context format (e.g., module-1-introduction)
            if not re.match(r'^[a-zA-Z0-9_-]+$', v.replace('/', '-').replace(' ', '-')):
                raise ValueError('Module context contains invalid characters')
        return v

    @field_validator('selected_text')
    @classmethod
    def validate_selected_text(cls, v):
        if v is not None:
            # Ensure selected text is at least 20 characters as per requirements
            if len(v) < 20:
                raise ValueError('Selected text must be at least 20 characters long')
        return v

    model_config = {"extra": "ignore"}


class QuestionValidator:
    """
    Utility class for validating different types of questions
    """
    
    @staticmethod
    def is_valid_question(text: str) -> bool:
        """
        Check if the text appears to be a valid question
        """
        if not text or len(text.strip()) < 3:
            return False
        
        # Check if it's a yes/no question or starts with question words
        question_indicators = [
            r'\b(how|what|when|where|why|who|which|whose|whom)\b',
            r'\bcan|could|would|should|is|are|do|does|did|will|would|am|i\'m|i am)\b',
            r'\?$'  # Ends with question mark
        ]
        
        text_lower = text.lower().strip()
        
        # Check if it contains question indicators
        for pattern in question_indicators:
            if re.search(pattern, text_lower):
                return True
                
        # Even if it doesn't have typical question markers, 
        # it might still be a valid information request
        if len(text_lower.split()) > 2:  # At least a few words
            return True
        
        return False

    @staticmethod
    def validate_question_context(question: str, module_context: Optional[str] = None) -> Dict[str, Any]:
        """
        Validate if a question makes sense in the given module context
        """
        validation_result = {
            "is_valid": True,
            "suggestions": [],
            "warnings": [],
            "module_relevance_score": 0.0
        }
        
        if module_context:
            # In a real implementation, we would check if the question
            # relates to the content in the specified module
            # For now, we'll just return a basic validation
            validation_result["module_relevance_score"] = 0.5  # Placeholder
            
            # Add warning if question seems unrelated to module context
            module_keywords = module_context.replace("-", " ").replace("_", " ").split()
            question_lower = question.lower()
            
            if not any(keyword in question_lower for keyword in module_keywords):
                validation_result["warnings"].append(
                    f"Question may not be directly related to the current module: {module_context}"
                )
        
        return validation_result

    @staticmethod
    def classify_question_type(question: str) -> str:
        """
        Classify the type of question for routing purposes
        """
        question_lower = question.lower()
        
        # Definition questions
        definition_indicators = [
            'what is', 'define', 'definition of', 'meaning of', 'what does.*mean'
        ]
        for indicator in definition_indicators:
            if re.search(indicator, question_lower):
                return "definition"
        
        # Explanation questions
        explanation_indicators = [
            'how does', 'explain', 'why is', 'how to', 'process of'
        ]
        for indicator in explanation_indicators:
            if re.search(indicator, question_lower):
                return "explanation"
        
        # Comparison questions
        comparison_indicators = [
            'difference between', 'compare', 'similarities', 'contrast'
        ]
        for indicator in comparison_indicators:
            if re.search(indicator, question_lower):
                return "comparison"
        
        # Example questions
        example_indicators = [
            'example of', 'give an example', 'show me', 'like'
        ]
        for indicator in example_indicators:
            if re.search(indicator, question_lower):
                return "example"
        
        return "general"

    @staticmethod
    def check_question_quality(question: str) -> Dict[str, Any]:
        """
        Check the quality of a question
        """
        quality_metrics = {
            "clarity_score": 0.0,
            "specificity_score": 0.0,
            "answerability": True,
            "suggestions": []
        }
        
        # Basic clarity assessment
        words = question.split()
        if len(words) < 3:
            quality_metrics["clarity_score"] = 0.3
            quality_metrics["suggestions"].append("Try expanding your question with more details")
        elif len(words) < 10:
            quality_metrics["clarity_score"] = 0.6
        else:
            quality_metrics["clarity_score"] = 0.8
        
        # Check for specific technical terms that might indicate good specificity
        technical_indicators = [
            r'\b(ros|node|service|topic|action|publisher|subscriber|tf|transform|frame)\b',
            r'\b(gazebo|simulation|unity|isaac|nvidia)\b',
            r'\b(kinematics|dynamics|inverse|forward|jacobian|trajectory)\b'
        ]
        
        specific_terms = 0
        for indicator in technical_indicators:
            if re.search(indicator, question.lower()):
                specific_terms += 1
        
        if specific_terms > 0:
            quality_metrics["specificity_score"] = 0.7 + (0.1 * specific_terms)
        else:
            quality_metrics["specificity_score"] = 0.4
            quality_metrics["suggestions"].append("Including technical terms might help get a more accurate answer")
        
        return quality_metrics


def validate_chat_request(request: ChatQueryRequest) -> Dict[str, Any]:
    """
    Comprehensive validation of a chat request
    """
    validation_result = {
        "is_valid": True,
        "errors": [],
        "warnings": [],
        "question_analysis": {}
    }
    
    # Validate message
    if not request.message or len(request.message.strip()) == 0:
        validation_result["is_valid"] = False
        validation_result["errors"].append("Message cannot be empty")
    else:
        # Check if message appears to be a valid question
        if not QuestionValidator.is_valid_question(request.message):
            validation_result["warnings"].append("The input doesn't appear to be a question. You can still ask for information.")
    
    # Validate selected text if present
    if request.selected_text:
        if len(request.selected_text) < 20:
            validation_result["is_valid"] = False
            validation_result["errors"].append("Selected text must be at least 20 characters")
    
    # Analyze question type and quality if it's a valid message
    if validation_result["is_valid"] and request.message:
        validation_result["question_analysis"] = {
            "type": QuestionValidator.classify_question_type(request.message),
            "quality": QuestionValidator.check_question_quality(request.message),
            "context_validation": QuestionValidator.validate_question_context(
                request.message, request.module_context
            )
        }
    
    return validation_result


# Example usage function
def validate_request_and_respond(request: ChatQueryRequest) -> Dict[str, Any]:
    """
    Validate request and return appropriate response
    """
    validation = validate_chat_request(request)
    
    if not validation["is_valid"]:
        return {
            "valid": False,
            "errors": validation["errors"],
            "message": "Request validation failed"
        }
    
    return {
        "valid": True,
        "analysis": validation["question_analysis"],
        "message": "Request is valid"
    }


if __name__ == "__main__":
    # Example usage
    try:
        # Valid request
        req = ChatQueryRequest(
            message="How do ROS 2 nodes communicate with each other?",
            module_context="module-2-ros2"
        )
        print("Valid request created")
        
        # Invalid request
        try:
            invalid_req = ChatQueryRequest(
                message="",  # This will fail validation
                module_context="module-2-ros2"
            )
        except ValueError as e:
            print(f"Validation correctly caught invalid request: {e}")
    except Exception as e:
        print(f"Error in validation example: {e}")
    
    print("Validation utility ready")