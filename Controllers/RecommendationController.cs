using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;
using Microsoft.ML;
using SmartWebAppAPI.Entity.Dto.Recommendation;
using SmartWebAppAPI.Entity.Models;
using static SmartWebAppAPI.MLModel;

namespace SmartWebAppAPI.Controllers
{
  [Route("api/[controller]")]
  [ApiController]
  public class RecommendationController : ControllerBase
  {

    [HttpPost("predict")]
    public IActionResult Post([FromBody] RecommendationDto recommendationDto)
    {
      string modelPath = Path.GetFullPath("MLModel.mlnet");
      var mlContext = new MLContext();
      ITransformer mlModel = mlContext.Model.Load(modelPath, out var modelSchema);
      var prediction = mlContext.Model.CreatePredictionEngine<ModelInput, ModelOutput>(mlModel);

      var input = new ModelInput
      {
        Co_rafya = recommendationDto.Cografya,
        Bina_Mimari = recommendationDto.Mimari,
        Veri_Density = recommendationDto.Veriİletim,
        Yerle_im_Plan_ = recommendationDto.Yerlesim
      };

      var predictionResult = prediction.Predict(input);
      return Ok(predictionResult.PredictedLabel);
    }

  }
}
