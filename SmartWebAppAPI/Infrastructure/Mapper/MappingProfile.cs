﻿
using AutoMapper;
using SmartWebAppAPI.Entity.Dto;
using SmartWebAppAPI.Entity.Models;

namespace SmartWebAppAPI.Infrastructure.Mapper
{
    public class MappingProfile : Profile
    {

       public MappingProfile() {

            CreateMap<RegisterDto, User>();
        
        }

    }
}