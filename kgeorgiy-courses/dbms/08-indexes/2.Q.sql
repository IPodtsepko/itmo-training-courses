-- Статистический запрос
select
    avg(cast(Mark as double)) as AvgMark
from
    Students
    natural join Marks
where
    GroupId = (
        select
            GroupId
        from
            Groups
        where
            GroupName = :GroupName
    )
    and CourseId in (
        select
            CourseId
        from
            Courses
        where
            CourseName = :CourseName
    );
